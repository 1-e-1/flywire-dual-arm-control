"""Offline estimator comparison on retained frames; truth NEVER enters fit."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from astronaut_delivery.motion import vertices,world,transformed
from astronaut_delivery.simultaneous import TightBoundArm
from astronaut_delivery.robot_only_geometry import RobotOnlyGuard
from astronaut_delivery.vision import Observation,TargetEstimate
from astronaut_delivery.placement_vision import placement_measurements
from astronaut_delivery.d435i import load_profile


def audit(run,scene,frames):
    root=Path(run);stage=Usd.Stage.Open(scene);bars={}
    for prim in stage.Traverse():
        match=re.match(r'/World/Station/Operation/Slot(\d+)_',str(prim.GetPath()))
        if match and vertices(prim) is not None:bars.setdefault(match[1],[]).append(transformed(vertices(prim),world(prim)).mean(0))
    truth=np.array([np.mean(p,axis=0)[:2] for p in bars.values()])
    if not len(truth):raise ValueError('no audit-only marker truth')
    arms=[TightBoundArm(stage,s) for s in ('left','right')];masker=RobotOnlyGuard(stage,*arms);cfg=load_profile();noise=cfg['simulation_noise']
    observations=json.loads((root/'observations.json').read_text());rows=[]
    selected=[(i,o) for i,o in enumerate(observations) if o['integrated']][-frames:]
    for i,row in selected:
        with np.load(root/'raw'/row['file']) as z:
            depth=z['depth'];exact='depth_sigma' in z
            sigma=z['depth_sigma'] if exact else noise['floor_sigma_m']+depth**2*noise['disparity_sigma_px']/(z['K'][0,0]*cfg['stereo_baseline_m'])
            robot_mask=z['robot_mask'] if 'robot_mask' in z else (depth>0)&(masker.mask_self_returns(depth,z['K'],z['camera_world'],z['q'])==0)
            obs=Observation(z['rgb'],depth,z['K'],z['camera_world'],row['time_s'],robot_mask,
                z['color_valid'] if 'color_valid' in z else z['rgb'].sum(2)>0,sigma)
            previous=observations[max(0,i-1)]['targets']
            priors=[TargetEstimate(n,np.array(t['position']),np.array(t['covariance']),t['pixels'],t['stamp'],t['kind']) for n,t in previous.items() if t['kind']=='placement_marking']
            measured=placement_measurements(obs,priors=priors)
        rows.append(dict(file=row['file'],time_s=row['time_s'],exact_saved_sensor_arrays=exact,
            new_estimates=[dict(position=e.position.tolist(),pixels=e.pixels,sigma_m=float(np.sqrt(np.linalg.eigvalsh(e.covariance).max())),
                nearest_true_marking_xy_error_m=float(np.linalg.norm(truth-e.position[:2],axis=1).min())) for e in measured],
            old_fresh_estimates=[dict(name=n,sigma_m=float(np.sqrt(np.linalg.eigvalsh(t['covariance']).max())),
                nearest_true_marking_xy_error_m=float(np.linalg.norm(truth-np.array(t['position'])[:2],axis=1).min()))
                for n,t in row['targets'].items() if t['kind']=='placement_marking' and abs(t['stamp']-row['time_s'])<1e-8]))
    errors=[e['nearest_true_marking_xy_error_m'] for row in rows for e in row['new_estimates']]
    result=dict(scope='Retained full-frame development replay, not closed-loop new-policy evidence; truth used after estimation only',
        caveat='Old recordings omit sigma/self-mask arrays: sigma reconstructed from noisy depth and self mask from known robot FK; prior associations from previous image-derived estimates',
        frames=len(rows),estimates=len(errors),xy_error_p95_m=float(np.quantile(errors,.95)) if errors else None,
        xy_error_max_m=max(errors) if errors else None,rows=rows)
    output=root/'marking_edge_fit_audit.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--frames',type=int,default=20);a=p.parse_args()
    with threadpool_limits(1):audit(a.run,a.scene,a.frames)
