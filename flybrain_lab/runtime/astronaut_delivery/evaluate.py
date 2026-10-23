"""Post-run simulator-truth evaluation, isolated from image-only target guidance."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from .scene import ROOT,SCENE
from .motion import Arm,CollisionGuard

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(ROOT/'reports/astronaut_delivery_verified'));args=parser.parse_args()
    folder=Path(args.run);result=json.loads((folder/'result.json').read_text())
    rows=json.loads((folder/'trajectory.json').read_text());events=json.loads((folder/'events.json').read_text())
    cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    final=rows[-1];metrics={}
    for spec in cfg['parts']:
        pose=np.array(final['parts'][spec['name']]).T
        metrics[spec['name']]=dict(position_error_m=float(np.linalg.norm(pose[:3,3]-spec['destination'])),upright_error_rad=float(np.arccos(np.clip(pose[2,2],-1,1))))
    speed=0.
    for a,b in zip(rows[:-1],rows[1:]):
        dt=b['time_s']-a['time_s']
        if dt>1e-9:speed=max(speed,float(np.max(abs(np.array(b['q'])-a['q']))/dt))
    # Independent supervision audit: left's own planner need not optimize mutual
    # avoidance, but the stationary right arm must remain protected throughout.
    initial=next(e for e in events if e['event']=='planned' and e['phase']=='initial_view')
    assert initial['knots']==2,'This collapsed audit supports the direct initial survey only.'
    end=next(e['time_s'] for e in events if e['event']=='arrived' and e['phase']=='initial_view')
    survey=next(row for row in rows if abs(row['time_s']-end)<1e-8)
    with threadpool_limits(1):
        stage=Usd.Stage.Open(str(SCENE));left,right=Arm(stage,'left'),Arm(stage,'right')
        guard=CollisionGuard(stage,left,right)
        supervision=guard.edge('left',np.zeros(8),np.array(survey['q'][0]),np.zeros(8),continuous=True)
    ok=result['status']=='PASS' and result['continuous_audit'] and len(result['completed'])==3 and supervision and speed<=cfg['joint_speed_rad_s']+1e-8
    ok=ok and all(x['position_error_m']<=.018 for x in metrics.values())
    report=dict(status='PASS' if ok else 'FAIL',placement=metrics,capture_gaps_m={e['part']:e['capture_gap_m'] for e in events if e['event']=='attached'},max_logged_average_joint_speed_rad_s=speed,left_motion_right_supervision_continuous=supervision,
        note='Placement metrics use simulator truth for evaluation only; IK residuals are not real-world accuracy. Single fixed-layout deterministic test, not statistical success rate.')
    (folder/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if not ok:raise SystemExit(1)

if __name__=='__main__':main()
