"""Read-only diagnostics of sagittal symmetry in the assembled USD."""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from build_astronaut_dual_arm import USD, world, frame

s=Usd.Stage.Open(str(USD))
M=np.diag([1.,-1.,1.])
print('C world:',world(s.GetPrimAtPath('/World/C_structure')))
for i in range(1,9):
    stem='base/a1' if i==1 else f'a{i}'
    frames=[]
    for side in ('LeftArm','RightArm'):
        b=s.GetPrimAtPath(f'/World/{side}/{stem}')
        j=UsdPhysics.Joint(s.GetPrimAtPath(f'/World/{side}/{stem}/joint{i}'))
        frames.append(np.array(frame(j,0)*world(b)))
    l,r=frames
    print(i,'pivot_error',np.linalg.norm(l[3,:3]@M-r[3,:3]),'axis_dot',np.dot(l[2,:3]@M,r[2,:3]))
for side in ('LeftArm','RightArm'):
    print(side,'root',world(s.GetPrimAtPath('/World/'+side)))
    print('b8 visual',UsdGeom.Xformable(s.GetPrimAtPath(f'/World/{side}/b8/visuals')).GetLocalTransformation())
