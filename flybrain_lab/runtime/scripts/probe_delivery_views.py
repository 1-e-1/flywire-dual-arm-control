import numpy as np
from pxr import Usd
from astronaut_delivery.scene import SCENE
from astronaut_delivery.motion import Arm,CollisionGuard

s=Usd.Stage.Open(str(SCENE));l=Arm(s,'left');r=Arm(s,'right');g=CollisionGuard(s,l,r)
for target in ([.4,.37,1.48],[.6,.45,1.2],[.55,.55,1.1],[.6,.20,1.2],[.45,.6,1.4],[.35,.7,1.05],[.6,.4,1.45]):
    target=np.array(target);direction=np.array([.55,-.2,.815])-target;direction/=np.linalg.norm(direction)
    q=l.solve(target,direction,tries=10)
    print(target, 'IK fail' if q is None else g.check('left',q,np.zeros(8)),q,flush=True)
