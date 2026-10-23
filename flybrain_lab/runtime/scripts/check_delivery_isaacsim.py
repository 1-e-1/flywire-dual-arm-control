"""Native RTX workcell load/render smoke test, isolated from the kinematic runner.

Known local Kit shutdown hangs are bounded by a parent watchdog. It only ends
the child it created, preserves logs, and reports the engine warning explicitly.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def worker(folder):
    cache=ROOT/'logs/delivery_kit_cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ['OPTIX_CACHE_PATH']=str(cache/'optix')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':960,'height':720,'renderer':'RaytracedLighting','anti_aliasing':2,
        'extra_args':[f'--/app/tokens/cache={cache.as_posix()}',f'--/app/tokens/data={(cache/"data").as_posix()}',f'--/app/tokens/logs={(cache/"logs").as_posix()}']})
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Gf,Usd,UsdGeom
    from astronaut_kinematics import DualArmKinematics
    result=dict(status='FAIL',note='Native static RGB render only; trajectory safety is tested separately.')
    try:
        run_info=json.loads((folder/'result.json').read_text())
        omni.usd.get_context().open_stage(run_info.get('scene_path',str(ROOT/'isaacsim_model/astronaut_delivery_workcell.usd')))
        for _ in range(20):app.update()
        stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        robot=DualArmKinematics(stage)
        records=json.loads((folder/'trajectory.json').read_text())
        row=records[-1];robot.set_positions(*np.array(row['q']))
        for name,pose in row['parts'].items():
            UsdGeom.Xformable(stage.GetPrimAtPath('/World/Parts/'+name)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose))
        cam=stage.GetPrimAtPath('/World/OverviewCamera')
        UsdGeom.Xformable(cam).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.35,-2.45,1.95),Gf.Vec3d(.32,-.22,.95),Gf.Vec3d(0,0,1)).GetInverse())
        images={}
        sensor_profile=run_info.get('sensor_profile')
        left_path='/World/LeftArm/b8/D435i/ColorCamera' if sensor_profile else '/World/LeftArm/b8/Camera'
        for name,path in [('overview','/World/OverviewCamera'),('left_camera',left_path)]:
            size=(sensor_profile['color']['width'],sensor_profile['color']['height']) if sensor_profile and name=='left_camera' else (960,720)
            product=rep.create.render_product(path,size)
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product)
            for _ in range(8):rep.orchestrator.step(rt_subframes=2)
            pixels=np.asarray(annotator.get_data())
            assert pixels.shape[:2]==(size[1],size[0])
            spatial_std=float(pixels[:,:,:3].astype(float).mean(2).std())
            assert spatial_std>5,'Empty image'
            Image.fromarray(pixels).save(folder/f'isaacsim_{name}.png')
            images[name]=dict(shape=list(pixels.shape),spatial_std=spatial_std)
            annotator.detach(product);product.destroy()
        result.update(status='PASS',images=images)
    except Exception as exc:
        result.update(error=str(exc));raise
    finally:
        (folder/'native_render_worker.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('DELIVERY_NATIVE_RENDER '+json.dumps(result),flush=True)
        app.close(wait_for_replicator=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(ROOT/'reports/astronaut_delivery_verified'));parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();folder=Path(args.run).resolve();folder.mkdir(parents=True,exist_ok=True)
    if args.worker:return worker(folder)
    result_path=folder/'native_render_worker.json'
    if result_path.exists():raise RuntimeError('Use a fresh report directory or review the existing native render result before rerunning.')
    log_path=folder/'native_render.log';start=time.monotonic();saved=None;terminated=False
    with log_path.open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,__file__,'--worker','--run',str(folder)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        while child.poll() is None:
            if result_path.exists() and saved is None:saved=time.monotonic()
            if time.monotonic()-start>360 or (saved is not None and time.monotonic()-saved>30):
                child.terminate();child.wait(timeout=20);terminated=True;break
            time.sleep(.5)
    result=json.loads(result_path.read_text()) if result_path.exists() else dict(status='FAIL',error='Worker did not complete the render checks.')
    text=log_path.read_text(encoding='utf-8',errors='replace')
    result.update(worker_exit_code=child.returncode,watchdog_terminated_own_child=terminated,engine_mutex_warning='SharedMutex' in text)
    if result['status']=='PASS' and (terminated or child.returncode!=0 or result['engine_mutex_warning']):result['status']='PASS_WITH_ENGINE_WARNING'
    (folder/'native_render_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if result['status']=='FAIL':raise SystemExit(1)

if __name__=='__main__':main()
