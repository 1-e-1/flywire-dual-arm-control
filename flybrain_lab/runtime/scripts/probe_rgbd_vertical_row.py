"""Paired saved-map IK task-rank diagnostic. NEVER commands a robot."""
import argparse,json,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_rgbd_view_rollback import restore
from astronaut_delivery.source_freeze import fingerprint,unchanged


def probe(run,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    hashes=fingerprint(ROOT);t=restore(run)
    if t.transfer_phase not in ('lift','place'):raise ValueError('saved lift/place goal required')
    current=t.right.tcp(t.q[1]);delta=t.saved_placement_goal-current[:3,3]
    goal=current[:3,3]+delta*min(1.,.003/max(np.linalg.norm(delta),1e-12))
    env,_=t.make_environment(t.guide.estimates[t.active],0.);rows=[]
    for mode in ('fixed_rotation','vertical_row'):
        began=time.perf_counter();checks=[]
        def valid(q):
            pair=np.r_[t.q[0],q];certificate=env.segment(t.q.ravel(),pair)
            robot=t.self_guard.segment(t.q.ravel(),pair,held=t.held)
            checks.append(dict(environment=asdict(certificate),robot=bool(robot)))
            return certificate.allowed and robot
        options=dict(rotation=t.transport_rotation) if mode=='fixed_rotation' else dict(vertical_row=t.source_tcp[2,:3])
        q=t.right.solve(goal,[0,0,-1] if mode=='fixed_rotation' else None,seed=t.q[1],tries=8,
            max_delta=.30,valid=valid,rng=np.random.default_rng(4000+len(t.paths)),**options)
        rows.append(dict(mode=mode,candidate=q is not None,checks=checks,wall_s=time.perf_counter()-began,
            position_error_m=None if q is None else float(np.linalg.norm(t.right.tcp(q)[:3,3]-goal)),
            vertical_row_error=None if q is None else float(np.linalg.norm(t.right.tcp(q)[2,:3]-t.source_tcp[2,:3])),
            joints=None if q is None else q.tolist()))
    result=dict(scope='SAVED-MAP IK AND WHOLE-SWEEP GEOMETRY ONLY; NOT EXECUTED; no live clock/visual certification',
        run=str(run),phase=t.transfer_phase,remaining_goal=t.saved_placement_goal.tolist(),next_goal=goal.tolist(),
        source_unchanged=unchanged(ROOT,hashes),source_hashes=hashes,rows=rows)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.output)
