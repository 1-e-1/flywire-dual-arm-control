"""Isolated latest-frame RGB-D/map process; no shared mutable USD stage or GIL.

Input poses belong ONLY to the simulator renderer. Perception receives rendered
RGB/depth/K/extrinsics, never part poses or instance segmentation identifiers.
"""
import multiprocessing as mp
import queue,time,traceback,json
from pathlib import Path
import numpy as np

def latest_put(channel,value,blocking=True):
    while True:
        try:channel.put(value,block=blocking,timeout=.005 if blocking else None);return True
        except queue.Full:
            try:channel.get_nowait()
            except queue.Empty:
                if not blocking:return False

def vision_process(requests,results,folder,seed,device,scene_path=None,task_config=None,telemetry=False):
    from pxr import Usd,UsdGeom,Gf
    from threadpoolctl import threadpool_limits
    from .online_scene import ONLINE_SCENE
    from .scene import ROOT
    from .run import DualArmKinematics
    from .motion import Arm
    from .d435i import D435iCamera,load_profile
    from .vision import VisualGuide
    import cv2
    try:
        with threadpool_limits(1):
            stage=Usd.Stage.Open(str(scene_path or ONLINE_SCENE));stage.SetEditTarget(stage.GetSessionLayer())
            robot=DualArmKinematics(stage);left=Arm(stage,'left');right=Arm(stage,'right');camera=D435iCamera(device)
            cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
            if task_config is not None:cfg.update(task_config)
            from .telemetry import TelemetryRecorder
            recorder=TelemetryRecorder(folder) if telemetry else None
            coop=json.loads((ROOT/'configs/cooperative_delivery.json').read_text())
            guide=VisualGuide({p['name']:p['color'] for p in cfg['parts']},sensor_profile=load_profile());guide.reacquire_stale=True;guide.fit_circular_cap=True
            guide.max_age=coop['maximum_observation_age_s'];guide.docking_cap_radius=coop['docking_cap_radius_m']
            from .map_maintenance import replay_changes
            epoch=0;rows=[];moving_updates=0;map_changes=[];local_policy=False
            while True:
                request=requests.get()
                if request['kind']=='finish':
                    if local_policy:
                        from .measured_surface import export_surface
                        export_surface(guide.mapper,folder)
                    if recorder is not None:recorder.close()
                    surface,_=guide.mapper.surface();np.savez_compressed(Path(folder)/'measured_map.npz',points=surface.cpu().numpy())
                    (Path(folder)/'async_sensor_frames.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                    latest_put(results,dict(kind='finished',map_points=len(surface),moving_updates=moving_updates,map_changes=map_changes));return
                start=time.perf_counter();q=np.asarray(request['q']).reshape(2,8);robot.set_positions(*q)
                local_policy=request.get('map_policy')=='local'
                for name,pose in request['parts'].items():
                    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Parts/'+name)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose))
                epoch,applied=replay_changes(guide,epoch,request['epoch'],request.get('map_changes',[]))
                map_changes.extend(applied)
                if request.get('focus') is not None and not request.get('full',True):
                    obs=camera.capture_roi(stage,left.camera(q[0]),request['time_s'],request['focus'],seed=seed+request['id'])
                else:obs=camera.capture(stage,left.camera(q[0]),request['time_s'],seed=seed+request['id'])
                held=request['held']
                if held:
                    paint=np.array(guide.palette[held]);paint/=paint.sum()
                    chroma=obs.rgb/np.maximum(obs.rgb.sum(2,keepdims=True),1e-6)
                    obs.robot_mask|=np.linalg.norm(chroma-paint,axis=2)<.06
                    if request.get('map_policy')=='local':
                        from .map_maintenance import mask_carried_depth
                        mask_carried_depth(obs,right.tcp(q[1])[:3,3])
                before=guide.map_updates;guide.update(obs,integrate_map=True)
                if recorder is not None:recorder.record(obs,guide,request.get('active_target'),q,request.get('phase','unknown'),epoch)
                if request['moving']:moving_updates+=guide.map_updates-before
                row=dict(id=request['id'],exposure_s=request['time_s'],wall_s=time.perf_counter()-start,epoch=epoch,
                         targets={n:dict(position=e.position.tolist(),stamp=e.stamp,pixels=e.pixels) for n,e in guide.estimates.items()},map_updates=guide.map_updates)
                rows.append(row)
                # Small display thumbnail; all estimation above uses native grids.
                rgb=(cv2.resize(obs.rgb,(212,120))*255).astype(np.uint8);depth=cv2.resize(obs.depth,(212,120),interpolation=cv2.INTER_NEAREST)
                robot_mask=cv2.resize(obs.robot_mask.astype(np.uint8),(212,120),interpolation=cv2.INTER_NEAREST).astype(bool)
                latest_put(results,dict(kind='observation',id=request['id'],epoch=epoch,time_s=obs.stamp,estimates=guide.estimates,
                           frames=guide.frames,map_updates=guide.map_updates,moving_updates=moving_updates,quality=camera.last_quality,
                           rgb=rgb,depth=depth,robot_mask=robot_mask,wall_s=row['wall_s']))
    except Exception:
        latest_put(results,dict(kind='error',error=traceback.format_exc()))

class VisionService:
    def __init__(self,folder,seed,device='cuda:0',scene_path=None,task_config=None,telemetry=False):
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.results=context.Queue(maxsize=2)
        self.process=context.Process(target=vision_process,args=(self.requests,self.results,str(folder),seed,device,scene_path,task_config,telemetry),daemon=True)
        self.process.start();self.serial=0;self.last_id=-1;self.closed=False
    def submit(self,blocking=True,**state):
        self.serial+=1;latest_put(self.requests,dict(kind='observe',id=self.serial,**state),blocking);return self.serial
    def poll(self):
        latest=None
        while True:
            try:data=self.results.get_nowait()
            except queue.Empty:break
            if data['kind']=='error':raise RuntimeError(data['error'])
            if data['kind']=='observation' and data['id']>self.last_id:self.last_id=data['id'];latest=data
        return latest
    def wait(self,request_id,timeout=30.):
        deadline=time.perf_counter()+timeout
        while time.perf_counter()<deadline:
            try:data=self.results.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive():raise RuntimeError('Vision process exited')
                continue
            if data['kind']=='error':raise RuntimeError(data['error'])
            if data['kind']=='observation' and data['id']>=request_id:self.last_id=data['id'];return data
        raise TimeoutError('Vision observation timeout')
    def close(self):
        if self.closed:return {}
        self.closed=True
        if not self.process.is_alive():raise RuntimeError('Vision worker already exited before clean shutdown')
        latest_put(self.requests,dict(kind='finish'));deadline=time.perf_counter()+30;result={}
        while time.perf_counter()<deadline:
            try:data=self.results.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive():break
                continue
            if data['kind']=='finished':result=data;break
            if data['kind']=='error':result=data;break
        self.process.join(timeout=2)
        if self.process.is_alive():
            # Only this object's own isolated worker, never an external process.
            self.process.terminate();self.process.join(timeout=2)
            raise TimeoutError('Owned vision process failed to close')
        if result.get('kind')!='finished':raise RuntimeError('Vision worker did not finish cleanly: '+str(result))
        return result
