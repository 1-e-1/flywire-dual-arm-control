"""Independent truth payload audit; never imported by the RGB-D planner."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from threadpoolctl import threadpool_limits
from .motion import Payload,Hull,world
from .simultaneous import TightBoundArm
from .separation_recovery_delivery import BoundedSeparationGuard,RetrySeparation
from .reaction import CombinedReaction


def audit(run,scene):
    root=Path(run);result=json.loads((root/'result.json').read_text());paths=json.loads((root/'paths.json').read_text())
    steps=json.loads((root/'steps.json').read_text());events=result.get('transfer_events',[])
    attach=next((e for e in events if e['event']=='measured_payload_attached'),None)
    if attach is None:raise ValueError('no measured payload attachment to audit')
    if sum(e['event']=='measured_payload_attached' for e in events)!=1:raise ValueError('audit currently covers exactly one payload')
    stage=Usd.Stage.Open(scene);arms=[TightBoundArm(stage,s) for s in ('left','right')]
    truth=BoundedSeparationGuard(stage,*arms,.003);guard=RetrySeparation(truth);reaction=CombinedReaction(stage,*arms)
    T0=np.array(attach['tcp']);name=attach['name'];world_points=truth.parts[name].points
    local=(world_points-T0[:3,3])@T0[:3,:3];payload=Payload(name,local,np.eye(4))
    measured=np.load(root/'captured_measured_envelope.npz');equations=measured['equations']
    signed=world_points@equations[:,:3].T+equations[:,3];contains=(signed<=1e-8).all(1)
    rows=[]
    release=next((e for e in events if e['event']=='measured_release_command'),None)
    for path in paths:
        carried='payload_local_points' in path
        # A released body stays at its RELEASE transform, not the final TCP.
        after_release=release is not None and path['time_s']>=release['time_s']-1e-8
        if after_release:
            Tr=np.array(release['tcp'])
            truth.parts[name]=Hull.make('part/'+name,local@Tr[:3,:3].T+Tr[:3,3])
            truth.separation_axes.clear()
        options=dict(held=payload,support_contact=path.get('payload_contact_phase',path.get('phase')) in ('lift','place')) if carried else {}
        a,b=np.array(path['start']),np.array(path['end']);passed=guard.segment(a,b,**options)
        rows.append(dict(time_s=path['time_s'],phase=path.get('phase','approach'),payload=carried,allowed=passed))
    errors=[]
    for step in steps:
        s=step.get('wrench_state')
        if s is not None:errors.append(float(np.max(abs(reaction.reference_wrench(np.array(s['q']),np.array(s['v']),np.array(s['a']))-step['wrench']))))
    q=np.array(result['final_joints']);T=np.array(release['tcp']) if release is not None else arms[1].tcp(q[1])
    final_points=local@T[:3,:3].T+T[:3,3]
    placement=None
    if release is not None:
        markings={}
        for prim in stage.GetPrimAtPath('/World/Station/Operation').GetChildren():
            label=prim.GetName()
            if label.startswith('Slot'):
                key=label.split('_')[0];markings.setdefault(key,[]).append(np.array(world(prim)).T[:3,3])
        observed=np.array(release['measured_destination']);centers=[np.mean(v,axis=0) for v in markings.values()]
        if centers:
            center=min(centers,key=lambda c:np.linalg.norm(c[:2]-observed[:2]))
            prim=stage.GetPrimAtPath('/World/Parts/'+name);pose=np.array(world(prim)).T
            local_cap=np.array(prim.GetAttribute('magnet:contactLocal').Get());cap=(pose@np.r_[local_cap,1])[:3]
            cap_local=(cap-T0[:3,3])@T0[:3,:3];final_cap=cap_local@T[:3,:3].T+T[:3,3]
            pad=stage.GetPrimAtPath('/World/Station/Operation/Pad');pad_z=np.array(world(pad))[3,2]+UsdGeom.Cube(pad).GetSizeAttr().Get()*.5*np.array(world(pad))[2,2]
            placement=dict(true_horizontal_interface_error_m=float(np.linalg.norm(final_cap[:2]-center[:2])),
                true_bottom_minus_support_m=float(final_points[:,2].min()-pad_z),actual_release_observed=True)
    placement_passed=bool(placement is not None and placement['true_horizontal_interface_error_m']<=.010 and -.0025<=placement['true_bottom_minus_support_m']<=.008)
    released=any(e['command']=='release' and not e['feedback']['attached'] for e in json.loads((root/'actuation_events.json').read_text()))
    terminal_rest=bool(steps and max(np.linalg.norm(steps[-1]['v']),np.linalg.norm(steps[-1]['a']))<1e-6)
    single_passed=bool(release) and released and terminal_rest and placement_passed and all(r['allowed'] for r in rows) and bool(contains.all())
    out=dict(scope='Independent known simulation collision solids, actual payload CAD and Newton-Euler audit; no planner feedback',
        paths=rows,all_continuous_geometry_passed=bool(rows) and all(r['allowed'] for r in rows),
        actual_payload_vertices=len(world_points),measured_envelope_contains_actual_payload=bool(contains.all()),
        maximum_payload_support_violation_m=float(signed.max()),payload_paths_checked=sum(r['payload'] for r in rows),
        wrench_samples_checked=len(errors),all_logged_wrenches_checked=len(errors)==len(steps),
        maximum_logged_wrench_reference_error=max(errors) if errors else None,placement=placement,
        single_transfer_audited=single_passed,full_transfer_audited=single_passed and result['full_transfer_completed'],
        post_release_retreat_audited=single_passed and bool(result.get('post_release_retreat_completed')) and any(p['time_s']>=release['time_s']-1e-8 for p in paths),
        terminal_C2_rest=terminal_rest,placement_passed=placement_passed,
        reaction_model=reaction.manifest(),limitations=['Imported collision solids, not hardware metrology','External contact impulses are excluded','No post-release retreat audit unless logged'])
    (root/'transfer_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);a=p.parse_args()
    with threadpool_limits(1):audit(a.run,a.scene)
