"""Offline commissioning posture selection from saved RGB-D evidence.

Not a startup trajectory. Frozen-clock queries are legitimate only for this
offline analysis; a live runner must replay raw data and revalidate the result.
"""
import argparse,json,time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pxr import Usd
from threadpoolctl import threadpool_limits
from classical_fusion.planning_map import PlanningMapConfig,PlanningSnapshot
from classical_fusion.rgbd_guard import RGBDHullEnvironmentGuard
from .simultaneous import TightBoundArm,SimultaneousGuard
from .robot_only_geometry import RobotOnlyGuard


def search(run,scene,output,tries=40):
    destination=Path(output)
    if destination.exists():raise ValueError('preserve existing report')
    root=Path(run);report=json.loads((root/'result.json').read_text());z=np.load(root/'planning_map_state.npz')
    cfg=PlanningMapConfig(**report['map_config'])
    free=(z['free_views']>=cfg.minimum_free_views)&~z['surface_veto']&((z['weight']==0)|(z['distance']>0))
    mapper=SimpleNamespace(cfg=cfg,revision=0,latest_observation_time=time.monotonic(),lower=torch.from_numpy(z['lower']),spacing=torch.from_numpy(z['spacing']),shape=free.shape)
    snapshot=PlanningSnapshot(mapper,free);stage=Usd.Stage.Open(str(scene))
    arms=[TightBoundArm(stage,s) for s in ('left','right')];self_guard=RobotOnlyGuard(stage,*arms)
    env=RGBDHullEnvironmentGuard(*arms,snapshot);q=np.array(report['initial_joints']);reasons=Counter();rows=[]
    def valid(value):
        candidate=q.copy();candidate[1]=value
        ok,why=SimultaneousGuard(self_guard).box(candidate.ravel(),candidate.ravel())
        if not ok:reasons['robot/'+why]+=1;return False
        cert=env.segment(candidate.ravel(),candidate.ravel(),now_s=snapshot.built_at)
        reasons[cert.reason]+=1
        if not cert.allowed and env.last_rejection:rows.append(env.last_rejection)
        return cert.allowed
    rng=np.random.default_rng(9312);ready=None
    for position in ([.45,-.5,1.10],[.55,-.45,1.05],[.5,-.25,1.1],[.65,-.4,1.1],[.35,-.45,1.15]):
        ready=arms[1].solve(position,[0,0,-1.],seed=q[1],tries=tries,rng=rng,valid=valid)
        print(json.dumps(dict(position=position,found=ready is not None,reasons=reasons)),flush=True)
        if ready is not None:q[1]=ready;break
    out=dict(scope='Offline RGB-D commissioning INITIAL CONDITION, not executed trajectory; must revalidate after raw replay',
        allowed=ready is not None,initial_joints=q.tolist() if ready is not None else None,
        reasons=dict(reasons),rejections=rows[-20:],source_run=str(root),scene=str(scene))
    destination.write_text(json.dumps(out,indent=2),encoding='utf-8');return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--tries',type=int,default=40);a=p.parse_args()
    with threadpool_limits(1):search(a.run,a.scene,a.output,a.tries)


if __name__=='__main__':main()
