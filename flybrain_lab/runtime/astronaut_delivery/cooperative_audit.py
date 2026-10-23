"""Independent re-check of logged complete missions; never trusts PASS strings."""
import argparse,json,time
from pathlib import Path
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from .motion import CollisionGuard,Payload,Hull,world
from .simultaneous import SimultaneousGuard,TightBoundArm
from .reaction import CombinedReaction
from .smooth_path import SmoothPath


class AuditCollisionGuard(CollisionGuard):
    def separated(self,key,va,na,vb,nb,margin):
        if super().separated(key,va,na,vb,nb,margin):return True
        from .convex_separation import certified_axis
        axis=certified_axis(va,vb,margin)
        if axis is None:return False
        self.separation_axes[key]=axis
        return True

def audit(folder,output_name='independent_audit.json'):
    folder=Path(folder);load=lambda n:json.loads((folder/n).read_text())
    report=load('result.json');paths=load('certified_paths.json');records=load('trajectory.json');events=load('events.json');steps=load('control_steps.json')
    stage=Usd.Stage.Open(report['scene_path']);stage.SetEditTarget(stage.GetSessionLayer())
    left,right=TightBoundArm(stage,'left'),TightBoundArm(stage,'right');guard=AuditCollisionGuard(stage,left,right);paired=SimultaneousGuard(guard);model=CombinedReaction(stage,left,right)
    initial={p.GetName():np.array(world(p)).T for p in stage.GetPrimAtPath('/World/Parts').GetChildren()}
    local={n:(h.points-initial[n][:3,3])@initial[n][:3,:3] for n,h in guard.parts.items()}
    certificates=[e for e in events if e['event']=='joint_segment_certified'];failures=[];checked=0;start=time.perf_counter()
    if len(certificates)!=len(paths):failures.append('certificate_event_count')
    for index,(path,event) in enumerate(zip(paths,certificates)):
        stamp=event['time_s'];state=[r for r in records if r['time_s']<=stamp+1e-8][-1]
        for name,pose in state['parts'].items():
            t=np.array(pose).T;guard.parts[name]=Hull.make('part/'+name,local[name]@t[:3,:3].T+t[:3,3])
        payload=None
        if path['held']:
            name=path['held'];held=[r for r in records if r['held']==name][0]
            tcp=right.tcp(np.array(held['q'])[1]);pose=np.array(held['parts'][name]).T
            rel=np.linalg.inv(tcp)@pose
            payload=Payload(name,local[name]@rel[:3,:3].T+rel[:3,3],rel)
        if not paired.segment(path['start'],path['end'],held=payload,contact_part=path['contact_part'],support_contact=path['support_contact']):failures.append('continuous_geometry/'+str(index))
        T=path['duration_s'];motion=SmoothPath(np.array(path['start']),np.array(path['end']),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))
        if np.any(np.array(motion.bounds())>np.array([.3,.5,5.])+1e-7):failures.append('joint_derivative_limits/'+str(index))
        for step in steps:
            if not stamp+1e-8<step['time_s']<=stamp+T+1e-8:continue
            if step['phase']=='safety_brake':continue
            expected=motion.state(step['time_s']-stamp)
            for key,value in zip(('q','v','a','jerk'),expected):
                if not np.allclose(step[key],value,atol=1e-7,rtol=1e-6):failures.append('logged_curve/'+str(index)+'/'+key);break
        checked+=1
        if checked%10==0:print('audited_segments',checked,flush=True)
    # Cross-check momentum-derivative estimates with an independent Newton-Euler
    # computation at midpoint states reconstructed from polynomial trajectories.
    errors=[]
    for index in np.unique(np.linspace(0,len(steps)-1,min(24,len(steps))).astype(int)):
        step=steps[index]
        if step['phase']=='safety_brake':continue
        t=step['time_s']-.5*step['dt_s']
        active=[(p,e) for p,e in zip(paths,certificates) if e['time_s']-1e-8<=t<=e['time_s']+p['duration_s']+1e-8]
        if not active:failures.append('missing_midpoint');continue
        path,event=active[0];T=path['duration_s']
        motion=SmoothPath(np.array(path['start']),np.array(path['end']),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))
        q,v,a,_=motion.state(t-event['time_s']);errors.append(float(np.max(abs(model.reference_wrench(q,v,a)-step['wrench']))))
    if errors and max(errors)>1e-5:failures.append('independent_reaction_disagreement')
    captures=[e['capture_gap_m'] for e in events if e['event']=='attached']
    if not captures or max(captures)>.008:failures.append('capture_gap')
    if report['completed']!=['bolt','washer','bracket']:failures.append('not_all_three_delivered')
    from .scene import ROOT
    cfg=report.get('config') or json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    placement={p['name']:float(np.linalg.norm(np.array(records[-1]['parts'][p['name']]).T[:3,3]-p['destination'])) for p in cfg['parts']}
    if max(placement.values())>.003:failures.append('final_placement_error')
    if np.linalg.norm(steps[-1]['v'])>1e-8 or np.linalg.norm(steps[-1]['a'])>1e-8:failures.append('nonzero_terminal_motion')
    result=dict(status='PASS' if not failures else 'FAIL',failures=failures,continuously_rechecked_segments=checked,
                separation_search='face normals + verified cached axis + independent convex-distance axis proposal; unchanged margins',
                independent_wrench_max_abs_error=max(errors) if errors else None,capture_gaps_m=captures,
                final_placement_errors_m=placement,
                audit_wall_s=time.perf_counter()-start,
                scope='Known-CAD continuous paths, logged quintic states, analytic derivative limits, independent reaction, full delivery; not a hardware certificate')
    if Path(output_name).name!=output_name:raise ValueError('audit output must be a filename')
    (folder/output_name).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output-name',default='independent_audit.json');args=p.parse_args()
    with threadpool_limits(1):result=audit(args.run,args.output_name)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
