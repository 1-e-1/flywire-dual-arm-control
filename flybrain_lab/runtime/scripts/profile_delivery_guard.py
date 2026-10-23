"""Bounded diagnostic using an already logged transfer/place segment."""
import sys,json,time,cProfile,pstats
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from astronaut_delivery.motion import Arm,CollisionGuard,Payload
from astronaut_delivery.scene import SCENE
rows=json.loads(Path('reports/astronaut_delivery/trajectory.json').read_text())
a=[r for r in rows if r['phase']=='bolt/transfer'][-1]
b=[r for r in rows if r['phase']=='bolt/place'][-1]
with threadpool_limits(1):
    stage=Usd.Stage.Open(str(SCENE));left,right=Arm(stage,'left'),Arm(stage,'right');guard=CollisionGuard(stage,left,right)
    qa,qb=np.array(a['q']),np.array(b['q']);tcp=right.tcp(qa[1]);part=np.array(a['parts']['bolt']).T
    points=guard.parts['bolt'].points-[.58,-.5,.762]
    world_points=points@part[:3,:3].T+part[:3,3]
    payload=Payload('bolt',(world_points-tcp[:3,3])@tcp[:3,:3],np.linalg.inv(tcp)@part)
    profile=cProfile.Profile();profile.enable();start=time.perf_counter()
    ok=guard.edge('right',qa[1],qb[1],qa[0],continuous=True,held=payload,support_contact=True)
    profile.disable();print('certified',ok,'seconds',time.perf_counter()-start,'checks',guard.checks,flush=True)
    pstats.Stats(profile).sort_stats('cumtime').print_stats(12)
