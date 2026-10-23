"""Ordered saved-state exact-query reordering microbenchmark, no actuation."""
import argparse,json,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_rgbd_view_rollback import restore
from astronaut_delivery.vision import TargetEstimate


def evaluate(run,output):
    target=Path(output)
    if target.exists():raise FileExistsError(target)
    rows=[]
    for prefilter in (False,True):
        t=restore(run);original=t.make_environment;counts=[]
        def make(*args):
            env,snapshot=original(*args);segment=env.segment
            if not prefilter:env.fine.query_grid=lambda name:env.fine.snapshot.clearance_grid
            def wrapped(*values,**kw):
                result=segment(*values,**kw);counts.append(env.fine.cell_tests);return result
            env.segment=wrapped;return env,snapshot
        t.make_environment=make;visual=t.guide.estimates[t.active]
        waypoint=TargetEstimate('measured_waypoint',t.saved_placement_goal,visual.covariance,visual.pixels,visual.stamp)
        began=time.perf_counter();motion,why=t.plan_step(waypoint,0.,step_m=.003);elapsed=time.perf_counter()-began
        rows.append(dict(permission_prefilter=prefilter,wall_s=elapsed,candidate=motion is not None,reason=why,
            counted_forbidden_cell_tests=sum(counts),full_guard_calls=len(counts),
            rejected_reasons=dict(Counter(reason for d in t.decisions if d.get('status')=='IK_REJECTED' for reason in d['reasons'])),
            proposal=t.paths[-1] if motion is not None else None))
    out=dict(run=str(run),scope='Ordered saved-map planning microbenchmark; exact existing permission reordered; no execution or full-task speedup claim',
        original_cell_budget=1000000,query_mask_construction_max_cells=3000000,rows=rows)
    with target.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps({**out,'rows':[{k:v for k,v in r.items() if k!='proposal'} for r in rows]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):evaluate(a.run,a.output)
