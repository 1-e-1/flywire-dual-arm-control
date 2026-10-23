"""Independent multi-object geometry timeline audit, no planner import.

Scene object count is read ONLY HERE, after control has stopped. The output
distinguishes observed-queue exhaustion from actual scene completion.
"""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from threadpoolctl import threadpool_limits
from .motion import Payload,Hull,world
from .simultaneous import TightBoundArm
from .separation_recovery_delivery import BoundedSeparationGuard,RetrySeparation
from .reaction import CombinedReaction


def completed_scene(actual_names,validated_names,queue_exhausted):
    return bool(actual_names) and bool(queue_exhausted) and set(actual_names)==set(validated_names)


def retreat_evidence(result,completed,has_retreat,release_tcp,withdrawn_tcp):
    # A global legacy flag may belong to an EARLIER object in a mission.
    logged=bool(has_retreat and (completed is not None or
        ('mission_cycle_budget' not in result and result.get('post_release_retreat_completed',False))))
    displacement=np.asarray(withdrawn_tcp)[:3,3]-np.asarray(release_tcp)[:3,3]
    vertical=float(displacement[2]);horizontal=float(np.linalg.norm(displacement[:2]))
    return dict(retreat_logged=logged,retreat_vertical_displacement_m=vertical,
        retreat_horizontal_displacement_m=horizontal,
        retreat_geometry_completed=bool(logged and vertical>=.075 and horizontal<=.010))


def audit(run,scene):
    root=Path(run)
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    result=read('result.json');paths=read('paths.json');steps=read('steps.json');acts=read('actuation_events.json')
    events=result.get('transfer_events',[]);attaches=[e for e in events if e.get('event')=='measured_payload_attached']
    releases=[e for e in events if e.get('event')=='measured_release_command']
    names=[e['name'] for e in attaches]
    if len(set(names))!=len(names):raise ValueError('repeated attachment requires a different timeline model')
    stage=Usd.Stage.Open(scene);arms=[TightBoundArm(stage,s) for s in ('left','right')]
    truth=BoundedSeparationGuard(stage,*arms,.003);guard=RetrySeparation(truth);reaction=CombinedReaction(stage,*arms)
    actual_names=list(truth.parts);payloads={};objects=[]
    markings={}
    for prim in stage.GetPrimAtPath('/World/Station/Operation').GetChildren():
        if prim.GetName().startswith('Slot'):
            markings.setdefault(prim.GetName().split('_')[0],[]).append(np.array(world(prim)).T[:3,3])
    centers=[np.mean(v,axis=0) for v in markings.values()]
    pad=stage.GetPrimAtPath('/World/Station/Operation/Pad');P=np.array(world(pad))
    pad_z=P[3,2]+UsdGeom.Cube(pad).GetSizeAttr().Get()*.5*P[2,2]
    for attach in attaches:
        name=attach['name'];T0=np.array(attach['tcp']);points=truth.parts[name].points
        local=(points-T0[:3,3])@T0[:3,:3];payloads[name]=Payload(name,local,np.eye(4))
        with np.load(root/attach.get('envelope_file','captured_measured_envelope.npz')) as measured:
            equations=measured['equations'];signed=points@equations[:,:3].T+equations[:,3]
        release=next((e for e in releases if e['target']==name),None)
        row=dict(name=name,envelope_contains_actual=bool((signed<=1e-8).all()),maximum_envelope_violation_m=float(signed.max()),
            vertices=len(points),placement_passed=False,retreat_logged=False,retreat_geometry_completed=False,release_confirmed=False)
        if release is not None and centers:
            T=np.array(release['tcp']);final_points=local@T[:3,:3].T+T[:3,3]
            prim=stage.GetPrimAtPath('/World/Parts/'+name);pose=np.array(world(prim)).T
            cap=(pose@np.r_[np.array(prim.GetAttribute('magnet:contactLocal').Get()),1])[:3]
            final_cap=((cap-T0[:3,3])@T0[:3,:3])@T[:3,:3].T+T[:3,3]
            center=min(centers,key=lambda c:np.linalg.norm(c[:2]-np.array(release['measured_destination'])[:2]))
            error=float(np.linalg.norm(final_cap[:2]-center[:2]));gap=float(final_points[:,2].min()-pad_z)
            completed=next((e for e in events if e.get('event')=='observed_object_cycle_completed' and e['name']==name),None)
            upper=completed['time_s'] if completed else min([e['time_s'] for e in attaches if e['time_s']>release['time_s']]+[np.inf])
            retreat=[p for p in paths if p.get('phase')=='retreat' and release['time_s']-1e-8<=p['time_s']<upper]
            recorded=[s for s in steps if release['time_s']-1e-8<=s['time_s']<=upper+1e-8]
            # Use actual executed state, not the proposed endpoint of a path
            # that could have been interrupted by a visual-quality brake.
            withdrawn=arms[1].tcp(np.asarray(recorded[-1]['q']).reshape(2,8)[1]) if recorded else T
            row.update(horizontal_error_m=error,bottom_minus_support_m=gap,placement_passed=error<=.010 and -.0025<=gap<=.008,
                release_confirmed=any(a['command']=='release' and a.get('target')==name and not a['feedback']['attached'] for a in acts),
                **retreat_evidence(result,completed,bool(retreat),T,withdrawn))
        objects.append(row)
    rows=[];applied=set()
    for path in paths:
        for release in releases:
            name=release['target']
            if name not in applied and path['time_s']>=release['time_s']-1e-8:
                T=np.array(release['tcp']);local=payloads[name].local_points
                truth.parts[name]=Hull.make('part/'+name,local@T[:3,:3].T+T[:3,3]);applied.add(name)
                truth.separation_axes.clear()
        carried=path.get('payload_name') if 'payload_local_points' in path else None
        options={} if carried is None else dict(held=payloads[carried],support_contact=path.get('payload_contact_phase') in ('lift','place'))
        passed=guard.segment(np.array(path['start']),np.array(path['end']),**options)
        rows.append(dict(time_s=path['time_s'],phase=path.get('phase','approach'),payload=carried,allowed=passed))
    errors=[]
    for step in steps:
        state=step.get('wrench_state')
        if state is not None:errors.append(float(np.max(abs(reaction.reference_wrench(*[np.array(state[k]) for k in ('q','v','a')])-step['wrench']))))
    geometry=bool(rows) and all(r['allowed'] for r in rows)
    rest=bool(steps and max(np.linalg.norm(steps[-1]['v']),np.linalg.norm(steps[-1]['a']))<1e-6)
    wrench=bool(errors) and len(errors)==len(steps) and max(errors)<=1e-5
    validated=[r['name'] for r in objects if all(r[k] for k in ('envelope_contains_actual','placement_passed','retreat_geometry_completed','release_confirmed'))]
    complete=completed_scene(actual_names,validated,result.get('observed_queue_exhausted')) and geometry and rest and wrench and result.get('source_unchanged_during_run',False)
    out=dict(scope='Post-execution actual simulation geometry/count only; never passed to planner',actual_scene_object_count=len(actual_names),
        actual_scene_objects=actual_names,objects=objects,paths=rows,all_continuous_geometry_passed=geometry,terminal_C2_rest=rest,
        all_logged_wrenches_checked=len(errors)==len(steps),wrench_samples_checked=len(errors),
        maximum_logged_wrench_reference_error=max(errors) if errors else None,wrench_crosscheck_passed=wrench,
        audited_released_and_retreated_objects=validated,scene_all_objects_delivered_audited=bool(complete),
        controller_observed_queue_exhausted=result.get('observed_queue_exhausted',False),
        limitations=['Simulation collision solids, not hardware metrology','External contact impulses and excluded inertias remain excluded',
            'Queue exhaustion alone is not actual scene completion'])
    (root/'mission_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);a=p.parse_args()
    with threadpool_limits(1):audit(a.run,a.scene)
