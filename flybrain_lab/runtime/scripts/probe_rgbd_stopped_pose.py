"""Saved-map diagnostic only; never issues commands or supplies CAD obstacles."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from classical_fusion.rgbd_guard import RGBDChordEnvironmentGuard
from classical_fusion.local_tool_guard import SelectedRobotHulls
from astronaut_delivery.simultaneous import TightBoundArm


def probe(run,scene,stationary_only=False):
    root=Path(run);r=json.loads((root/'result.json').read_text());q=np.array(r['final_joints']).ravel()
    z=np.load(root/'planning_map_state.npz');meta=json.loads((root/'measured_surface.json').read_text())
    mapper=SimpleNamespace(_lock=RLock(),cfg=PlanningMapConfig(**meta['config']),revision=0,
        latest_observation_time=time.monotonic(),shape=z['distance'].shape,
        **{k:torch.from_numpy(z[k]) for k in ('distance','weight','free_views','surface_veto','lower','spacing')})
    snapshot=RobotQuerySnapshot(mapper);snapshot.max_age=1e9
    stage=Usd.Stage.Open(str(scene));left,right=[TightBoundArm(stage,s) for s in ('left','right')]
    env=RGBDChordEnvironmentGuard(left,SelectedRobotHulls(right,lambda n:n!='magnet'),snapshot,max_boxes=127,max_depth=8)
    print('stationary_box',env.box(q,q),env.last_rejection)
    print('stationary_chord',env.chord(q,q),env.last_rejection)
    print('stationary_segment',env.segment(q,q),env.last_rejection)
    if env.last_rejection:
        for index in env.last_rejection['voxel_indices']:
            key=tuple(index)
            print('rejected_cell',index,{name:z[name][key].item() for name in ('distance','weight','free_views','surface_veto')})
    from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
    interval=StaticIntervalSnapshot(mapper,assume_static_epoch=True);interval.max_age=1e9
    refined=RGBDChordEnvironmentGuard(left,SelectedRobotHulls(right,lambda n:n!='magnet'),interval,max_boxes=127,max_depth=8)
    print('static_interval_stationary_segment',refined.segment(q,q),refined.last_rejection)
    print('right_tcp',right.tcp(q[8:])[:3,3])
    if stationary_only:return
    paths=json.loads((root/'paths.json').read_text());goal=np.array(paths[-1]['target'])+[0,0,paths[-1]['standoff_m']];tcp=right.tcp(q[8:])
    goal=tcp[:3,3]+(goal-tcp[:3,3])*min(1.,.005/max(np.linalg.norm(goal-tcp[:3,3]),1e-9))
    hints=[]
    def valid(angles):
        cert=env.segment(q,np.r_[q[:8],angles])
        if not cert.allowed:hints.append(env.last_rejection)
        return cert.allowed
    candidate=right.solve(goal,[0,0,-1],seed=q[8:],rotation=tcp[:3,:3],max_delta=.30,tries=8,valid=valid)
    print('continuous_valid_IK',None if candidate is None else candidate.tolist())
    if candidate is None:
        from astronaut_delivery.evidence_ik import residual_from_rejections
        residual=residual_from_rejections(right,hints,start=q[8:])
        if residual is not None:candidate=right.solve(goal,[0,0,-1],seed=q[8:],rotation=tcp[:3,:3],max_delta=.30,tries=4,valid=valid,extra_residual=residual)
        print('evidence_guided_IK',None if candidate is None else candidate.tolist())
    if candidate is None:
        down=tcp[:3,3]-[0,0,.006]
        candidate=right.solve(down,[0,0,-1],seed=q[8:],rotation=tcp[:3,:3],max_delta=.30,tries=8,valid=valid)
        print('lower_6mm_broad_only',None if candidate is None else candidate.tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--stationary-only',action='store_true');a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.scene,a.stationary_only)
