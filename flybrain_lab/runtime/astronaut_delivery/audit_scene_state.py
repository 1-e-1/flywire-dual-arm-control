"""Post-hoc simulator truth transforms for map audits, never planner input."""
import numpy as np
from pxr import Gf,UsdGeom
from .motion import world


def moved_pose(original,capture_tcp,current_tcp):
    return np.asarray(current_tcp)@np.linalg.inv(np.asarray(capture_tcp))@np.asarray(original)


def apply_recorded_final_state(stage,result):
    from .simultaneous import TightBoundArm
    events=result.get('transfer_events',[]);attached=[e for e in events if e.get('event')=='measured_payload_attached']
    release=[e for e in events if e.get('event')=='measured_release_command'];applied=[]
    right=None
    for event in attached:
        name=event['name'];body=stage.GetPrimAtPath('/World/Parts/'+name)
        if not body:raise ValueError('audit object absent from scene: '+name)
        if name in applied:raise ValueError('repeated ownership requires chronological audit')
        final=next((e for e in release if e['target']==name),None)
        if final is not None:T=np.array(final['tcp'])
        else:
            if right is None:right=TightBoundArm(stage,'right')
            T=right.tcp(np.asarray(result['final_joints'])[1])
        transform=moved_pose(np.array(world(body)).T,np.array(event['tcp']),T)
        ops=UsdGeom.Xformable(body).GetOrderedXformOps()
        if len(ops)!=1 or ops[0].GetOpType()!=UsdGeom.XformOp.TypeTransform:raise ValueError('unsupported audit object transform')
        ops[0].Set(Gf.Matrix4d(transform.T.tolist()));applied.append(name)
    return applied
