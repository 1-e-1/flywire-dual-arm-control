"""Mass-property sensitivity on actual pure RGB-D logged midpoint states.

Open-loop engineering perturbations, not closed-loop robustness or hardware
uncertainty calibration. Every sample is retained; no frame subsampling.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pxr import Usd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astronaut_delivery.simultaneous import TightBoundArm
from astronaut_delivery.reaction import CombinedReaction,reaction_metrics
from astronaut_delivery.inertia_sensitivity import perturb_composites


def evaluate(run,scene,draws=32,seed=20260910):
    root=Path(run);result=json.loads((root/'result.json').read_text())
    if result.get('environment_authority')!='RGBD_ONLY':raise ValueError('pure RGB-D run required')
    steps=json.loads((root/'steps.json').read_text())
    if not steps or not all('wrench_state' in s for s in steps):raise ValueError('complete logged midpoint states required')
    q,v,a=[np.array([s['wrench_state'][k] for s in steps]) for k in ('q','v','a')]
    dt=np.array([s['dt_s'] for s in steps]);logged=np.array([s['wrench'] for s in steps])
    stage=Usd.Stage.Open(str(scene));model=CombinedReaction(stage,*[TightBoundArm(stage,s) for s in ('left','right')])
    nominal_w=model.wrench_batch(q,v,a);error=float(np.max(abs(nominal_w-logged)))
    if error>1e-5:raise ValueError('nominal model disagrees with actual midpoint log')
    nominal=reaction_metrics(nominal_w,dt);rng=np.random.default_rng(seed);rows=[]
    originals=[copy.deepcopy(m.composites) for m in model.models]
    try:
        for draw in range(draws):
            for m in model.models:m.composites=perturb_composites(m,rng)
            rows.append(reaction_metrics(model.wrench_batch(q,v,a),dt))
    finally:
        for m,c in zip(model.models,originals):m.composites=c
    summary={}
    for kind in ('force','torque'):
        for metric in ('peak','rms','accumulated_magnitude','maximum_window_impulse'):
            values=np.array([r[kind][metric] for r in rows])
            summary[kind+'_'+metric]=dict(nominal=nominal[kind][metric],minimum=float(values.min()),maximum=float(values.max()),
                median=float(np.median(values)),q95=float(np.quantile(values,.95)))
    return dict(scope='Open-loop sensitivity of the same measured-only executed kinematics, not robustness certification',
        run=str(root),scene=str(scene),seed=seed,draws=draws,all_midpoint_samples=len(steps),nominal_log_max_error=error,
        mass_unchanged=True,com_axis_bound_m=.005,inertia_scale=[.8,1.2],inertia_axis_rotation_deg=5.,
        nominal=nominal,summary=summary,draw_metrics=rows,
        exclusions=['No sensor or controller rerun','No physical probability distribution','Payload/tool/camera inertia remains excluded'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=32);a=p.parse_args()
    if not 1<=a.draws<=1000:raise ValueError('bounded nonempty draw count required')
    target=Path(a.output)
    if target.exists():raise FileExistsError(target)
    with threadpool_limits(1):out=evaluate(a.run,a.scene,a.draws)
    target.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out['summary'],indent=2))
