"""Independent post-execution truth audit; NEVER a controller dependency."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from .simultaneous import TightBoundArm
from .separation_recovery_delivery import BoundedSeparationGuard,RetrySeparation
from .reaction import CombinedReaction
from .smooth_path import SmoothPath
from .motion import world


def audit(run,scene,target='bracket'):
    root=Path(run);result=json.loads((root/'result.json').read_text());paths=json.loads((root/'paths.json').read_text())
    stage=Usd.Stage.Open(str(scene));arms=[TightBoundArm(stage,s) for s in ('left','right')]
    geometry=RetrySeparation(BoundedSeparationGuard(stage,*arms,.003));reaction=CombinedReaction(stage,*arms)
    rows=[];errors=[];logged_errors=[]
    for path in paths:
        a,b=np.array(path['start']),np.array(path['end']);allowed=geometry.segment(a,b)
        T=path['duration_s'];motion=SmoothPath(a,b,T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))
        for t in np.linspace(0,T,9):
            q,v,acc,_=motion.state(t);errors.append(float(np.max(abs(reaction.wrench(q,v,acc)-reaction.reference_wrench(q,v,acc)))))
        rows.append(dict(start_time_s=path['time_s'],allowed=allowed))
    steps=json.loads((root/'steps.json').read_text())
    for step in steps:
        state=step.get('wrench_state')
        if state is None:continue
        reference=reaction.reference_wrench(np.array(state['q']),np.array(state['v']),np.array(state['a']))
        logged_errors.append(float(np.max(abs(reference-np.array(step['wrench'])))))
    # Truth is read only HERE, after the RGB-D controller has finished.
    part=stage.GetPrimAtPath('/World/Parts/'+target);pose=np.array(world(part)).T
    cap=(pose@np.r_[np.array(part.GetAttribute('magnet:contactLocal').Get()),1])[:3]
    q=np.array(result['final_joints']);tcp=arms[1].tcp(q[1])[:3,3]
    desired=cap+[0,0,paths[-1]['standoff_m']] if paths else cap
    measured=paths[-1]['target'] if paths else None
    out=dict(scope='Independent environment CAD truth and Newton-Euler audit only; no controller feedback',
        certified_paths=rows,all_continuous_geometry_passed=bool(rows) and all(x['allowed'] for x in rows),
        maximum_reaction_reference_error=float(max(errors)) if errors else None,
        executed_wrench_samples_checked=len(logged_errors),
        all_logged_wrenches_independently_checked=bool(steps) and len(logged_errors)==len(steps),
        maximum_logged_wrench_reference_error=max(logged_errors) if logged_errors else None,
        reaction_model=reaction.manifest(),
        final_true_capture_gap_m=float(np.linalg.norm(tcp-cap)),
        final_true_standoff_error_m=float(np.linalg.norm(tcp-desired)),
        final_measured_target_error_m=float(np.linalg.norm(np.array(measured)-cap)) if measured is not None else None,
        full_transfer_audited=False)
    (root/'execution_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2));return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--target',default='bracket');a=p.parse_args()
    with threadpool_limits(1):audit(a.run,a.scene,a.target)


if __name__=='__main__':main()
