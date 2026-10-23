import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from astronaut_delivery.motion import Arm,CollisionGuard
from astronaut_delivery.magnet import MagneticLatch
from collections import Counter

with threadpool_limits(1):
    stage=Usd.Stage.Open('reports/astronaut_delivery_probe/final_state.usdc')
    left,right=Arm(stage,'left'),Arm(stage,'right');guard=CollisionGuard(stage,left,right)
    q=np.array(json.loads(Path('reports/astronaut_delivery_probe/trajectory.json').read_text())[-1]['q'])
    latch=MagneticLatch(stage,guard);print(latch.attach('bolt',right.tcp(q[1])),flush=True)
    for target in ([.35,-.48,.84],[.35,-.34,.84],[.35,-.20,.84],[.35,-.48,.762],[.35,-.34,.762],[.35,-.20,.762]):
        reasons=Counter()
        def valid(x):
            ok,reason=guard.check('right',x,q[0],held=latch.payload)
            reasons[reason]+=1
            return ok
        sol=right.solve(target,[0,0,1],seed=q[1],offset=latch.payload.local_pose,tries=60,valid=valid)
        print(target,sol,reasons,flush=True)
