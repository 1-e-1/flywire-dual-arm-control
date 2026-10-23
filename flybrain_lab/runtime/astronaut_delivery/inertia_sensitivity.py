"""Open-loop mass-property uncertainty audit on an unchanged logged trajectory."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd
from .motion import Arm
from .online_scene import ONLINE_SCENE
from .reaction import CombinedReaction,reaction_metrics
from .smooth_path import SmoothPath


def logged_midpoints(root,steps):
    paths=json.loads((root/'certified_paths.json').read_text())
    events=[e for e in json.loads((root/'events.json').read_text()) if e['event']=='joint_segment_certified']
    if len(paths)!=len(events) or any(s['phase']=='safety_brake' for s in steps):raise ValueError('Expected unbraked complete mission')
    stamps=np.array([e['time_s'] for e in events]);curves=[]
    for path in paths:
        T=path['duration_s'];curves.append(SmoothPath(np.array(path['start']),np.array(path['end']),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5])))
    states=[]
    for s in steps:
        mid=s['time_s']-.5*s['dt_s'];i=int(np.searchsorted(stamps,mid,side='right')-1)
        if i<0 or not -1e-7<=mid-stamps[i]<=curves[i].duration+1e-7:raise ValueError('Missing trajectory interval')
        states.append(curves[i].state(mid-stamps[i])[:3])
    return tuple(np.array(states)[:,j,:] for j in range(3))


def perturb_composites(model,rng,com_bound=.005,inertia_fraction=.2):
    """Perturb each original module, then aggregate with parallel-axis theorem."""
    state=model.tree.evaluate(np.zeros(8));groups={}
    for name,m,c,I in model.bodies:
        delta=rng.uniform(-com_bound,com_bound,3)
        axis=rng.normal(size=3);axis/=max(np.linalg.norm(axis),1e-12)
        R=Rotation.from_rotvec(axis*rng.uniform(-np.deg2rad(5),np.deg2rad(5))).as_matrix()
        # Positive uniform scaling preserves principal-inertia triangle inequalities.
        J=(R@I@R.T)*rng.uniform(1-inertia_fraction,1+inertia_fraction)
        groups.setdefault(tuple(state.chains[name]),[]).append((name,m,c+delta,J))
    composites=[]
    for group in groups.values():
        anchor=group[0][0];inv=np.linalg.inv(state.poses[anchor]);items=[]
        for name,m,c,I in group:
            T=inv@state.poses[name];R=T[:3,:3];items.append((m,R@c+T[:3,3],R@I@R.T))
        mass=sum(m for m,c,I in items);center=sum(m*c for m,c,I in items)/mass
        inertia=sum(I+m*(np.dot(c-center,c-center)*np.eye(3)-np.outer(c-center,c-center)) for m,c,I in items)
        composites.append((anchor,mass,center,inertia))
    return composites


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--draws',type=int,default=100);args=p.parse_args()
    stage=Usd.Stage.Open(str(ONLINE_SCENE));model=CombinedReaction(stage,Arm(stage,'left'),Arm(stage,'right'))
    steps=json.loads((Path(args.run)/'control_steps.json').read_text())
    q,v,a=logged_midpoints(Path(args.run),steps);dt=np.array([s['dt_s'] for s in steps])
    nominal_w=model.wrench_batch(q,v,a);agreement=float(np.max(abs(nominal_w-np.asarray([s['wrench'] for s in steps]))))
    if agreement>1e-6:raise ValueError('Midpoint reaction disagrees with logged controller')
    nominal=reaction_metrics(nominal_w,dt);rng=np.random.default_rng(20260909);rows=[]
    originals=[copy.deepcopy(m.composites) for m in model.models]
    for i in range(args.draws):
        for m in model.models:m.composites=perturb_composites(m,rng)
        rows.append(reaction_metrics(model.wrench_batch(q,v,a),dt))
    for m,c in zip(model.models,originals):m.composites=c
    summary={}
    for kind in ('force','torque'):
        for metric in ('peak','rms','maximum_window_impulse'):
            values=np.array([r[kind][metric] for r in rows]);summary[kind+'_'+metric]=dict(
                nominal=nominal[kind][metric],perturbed_min=float(values.min()),perturbed_max=float(values.max()),
                perturbed_median=float(np.median(values)),perturbed_q95=float(np.quantile(values,.95)))
    out=dict(scope='Fixed logged kinematics, stochastic USD mass-property sensitivity only; neither robust closed-loop certification nor a calibrated physical distribution.',
        seed=20260909,draws=args.draws,midpoint_wrench_max_discrepancy=agreement,com_independent_axis_bound_m=.005,principal_inertia_scale_range=[.8,1.2],
        inertia_frame_rotation_max_deg=5,module_masses_unchanged=True,nominal=nominal,summary=summary,draw_metrics=rows)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
