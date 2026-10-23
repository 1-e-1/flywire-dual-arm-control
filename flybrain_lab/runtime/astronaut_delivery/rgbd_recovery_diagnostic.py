"""Read-only offline diagnostic of measured rescan candidates, NOT execution."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
import numpy as np
import torch
from pxr import Usd
from threadpoolctl import threadpool_limits
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from classical_fusion.rgbd_guard import RGBDChordEnvironmentGuard
from .robot_only_geometry import RobotOnlyGuard
from .simultaneous import TightBoundArm,SimultaneousGuard
from .reaction import CombinedReaction
from .visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from .continuous_visual import certify


def diagnose(run,scene):
    root=Path(run);result=json.loads((root/'result.json').read_text())
    z=np.load(root/'planning_map_state.npz');meta=json.loads((root/'measured_surface.json').read_text())
    mapper=SimpleNamespace(_lock=RLock(),cfg=PlanningMapConfig(**meta['config']),revision=0,
        latest_observation_time=time.monotonic(),shape=z['distance'].shape,
        **{k:torch.from_numpy(z[k]) for k in ('distance','weight','free_views','surface_veto','lower','spacing')})
    snapshot=RobotQuerySnapshot(mapper)
    # Freeze the clock ONLY for saved-data diagnostics. Never use this snapshot
    # to command hardware or the live simulation controller.
    snapshot.max_age=1e9
    stage=Usd.Stage.Open(str(scene));arms=[TightBoundArm(stage,s) for s in ('left','right')]
    env=RGBDChordEnvironmentGuard(*arms,snapshot,max_boxes=127,max_depth=8)
    guard=SimultaneousGuard(RobotOnlyGuard(stage,*arms));q=np.array(result['final_joints']);q0=q.ravel()
    target=json.loads((root/'observations.json').read_text())[-1]['targets']['bracket'];p=np.array(target['position'])
    allocator=VisualSetAllocator(arms[0],CombinedReaction(stage,*arms),SufficiencyConfig(objective='wrench',accept_feasible_iterate=True))
    context=allocator.context(q[0],p,np.eye(3)*(.003**2),target['pixels'],0.)
    initial=env.segment(q0,q0);print('CURRENT',initial,env.last_rejection,flush=True)
    camera=arms[0].camera(q[0])[:3,3];rows=[]
    for axis_mode in ('preserve','look_at'):
        for offset in ([0,0,.02],[0,0,.04],[0,0,.06],[.03,0,.02],[-.03,0,.02],[0,-.04,.02],[0,.04,.02]):
            position=camera+offset;axis=arms[0].tcp(q[0])[:3,2] if axis_mode=='preserve' else p-position;axis/=np.linalg.norm(axis)
            left=arms[0].solve(position,axis,seed=q[0],tries=4,rng=np.random.default_rng(71))
            row=dict(axis_mode=axis_mode,offset=offset,IK=left is not None)
            if left is not None:
                endpoint=np.r_[left,q[1]];c=env.segment(q0,endpoint)
                row.update(environment=c.reason,environment_allowed=c.allowed,rejection=env.last_rejection,
                    robot=guard.segment(q0,endpoint),visual=certify(allocator,q0[:8],left,context,max_boxes=1023).dictionary())
            rows.append(row);print(json.dumps(row),flush=True)
    (root/'recovery_diagnostic.json').write_text(json.dumps(dict(scope='Offline frozen-map diagnostic only',candidates=rows),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--scene',required=True);a=parser.parse_args()
    with threadpool_limits(1):diagnose(a.run,a.scene)
