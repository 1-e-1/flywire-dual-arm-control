"""Observed telemetry and periodic actual map snapshots, never hindsight maps."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

class TelemetryRecorder:
    def __init__(self,folder,period=2.):
        self.folder=Path(folder)/'telemetry';self.folder.mkdir(exist_ok=True)
        self.rows=[];self.last_snapshot=-np.inf;self.period=period;self.last_epoch=None
    def record(self,obs,guide,active_target,q,phase,epoch):
        estimates={n:dict(position=e.position.tolist(),covariance=e.covariance.tolist(),pixels=int(e.pixels),stamp=float(e.stamp)) for n,e in guide.estimates.items()}
        target=guide.target(active_target,obs.stamp) if active_target else None
        row=dict(exposure_s=float(obs.stamp),phase=phase,active_target=active_target,epoch=epoch,estimates=estimates,
                 q=np.asarray(q).tolist(),camera_world=obs.camera_world.tolist(),map_updates=guide.map_updates,
                 depth_valid_ratio=float(np.mean(obs.depth>0)),target_fresh=target is not None,frame_file=None,map_file=None)
        if target is not None:
            p=(target.position-obs.camera_world[:3,3])@obs.camera_world[:3,:3];z=-p[2]
            sigma=float(np.sqrt(np.linalg.eigvalsh(target.covariance).max()))
            row.update(range_m=float(np.linalg.norm(p)),normalized_uv=[float(p[0]/z),float(-p[1]/z)] if z>0 else None,
                       pixels=int(target.pixels),position_sigma_max_m=sigma,target_age_s=float(obs.stamp-target.stamp))
            row['diagnostic_quality_sufficient']=bool(z>0 and .45<=np.linalg.norm(p)<=1.2 and target.pixels>=12 and sigma<=.005 and abs(p[0]/z)<=.60 and abs(p[1]/z)<=.33)
        else:row['diagnostic_quality_sufficient']=False
        if obs.stamp-self.last_snapshot>=self.period or epoch!=self.last_epoch:
            index=len(self.rows);frame=f'frame_{index:05d}.png';cloud=f'map_{index:05d}.npz'
            Image.fromarray((np.clip(obs.rgb,0,1)*255).astype(np.uint8)).save(self.folder/frame)
            surface,_=guide.mapper.surface();points=surface.cpu().numpy()
            np.savez_compressed(self.folder/cloud,points=points)
            row.update(frame_file=frame,map_file=cloud,map_points=len(points));self.last_snapshot=obs.stamp;self.last_epoch=epoch
        self.rows.append(row)
    def close(self):
        (self.folder/'observations.json').write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
