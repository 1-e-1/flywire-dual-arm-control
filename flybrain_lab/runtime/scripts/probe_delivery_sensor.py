import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from pxr import Usd
from PIL import Image
from astronaut_delivery.scene import SCENE,ROOT
from astronaut_delivery.motion import Arm,CollisionGuard
from astronaut_delivery.vision import MeshCamera,VisualGuide
from astronaut_kinematics import DualArmKinematics

s=Usd.Stage.Open(str(SCENE));s.SetEditTarget(s.GetSessionLayer())
l=Arm(s,'left');r=Arm(s,'right');g=CollisionGuard(s,l,r)
target=np.array([.60,.45,1.20]);direction=np.array([.55,-.2,.815])-target;direction/=np.linalg.norm(direction)
q=l.solve(target,direction,tries=15,valid=lambda q:g.check('left',q,np.zeros(8))[0])
print('view',q,flush=True)
print('direct view edge',g.edge('left',np.zeros(8),q,np.zeros(8)),flush=True)
robot=DualArmKinematics(s);robot.set_positions(q,np.zeros(8))
camera=MeshCamera();obs=camera.capture(s,l.camera(q),0)
Image.fromarray((obs.rgb*255).astype('uint8')).save(ROOT/'isaacsim_model'/'delivery_camera_probe.png')
cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text());guide=VisualGuide({p['name']:p['color'] for p in cfg['parts']})
guide.update(obs)
print('targets',{k:v.position.tolist() for k,v in guide.estimates.items()},flush=True)
print('gt evaluation only',{p['name']:(np.array(p['position'])+[0,0,.055]).tolist() for p in cfg['parts']},flush=True)
