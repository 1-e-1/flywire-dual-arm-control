"""Fault injection into the actual streaming controller on recorded segments.

This is feedback/log replay, NOT new rendered closed-loop scene experiments.
Before a fault, measurements retain their recorded receive/exposure times.
After visual braking begins no new replayed image authorizes continued motion.
Geometry is inherited only from an independently audited unchanged segment.
"""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from astronaut_delivery.rgbd_cooperative import RGBDCooperative
from astronaut_delivery.smooth_path import SmoothPath
from astronaut_delivery.simultaneous import TightBoundArm
from astronaut_delivery.reaction import CombinedReaction,reaction_metrics
from astronaut_delivery.vision import TargetEstimate
from astronaut_delivery.visual_sufficiency import SufficiencyConfig


def replay_case(path,observations,model,fault,fraction):
    start=float(path['time_s']);T=float(path['duration_s']);onset=start+fraction*T
    curve=SmoothPath(np.array(path['start']),np.array(path['end']),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))
    run=RGBDCooperative.__new__(RGBDCooperative);run.q=curve.start.reshape(2,8).copy();run.time=start
    run.steps=[];run.active=path['visual_target_name'];run.reaction=model
    run.allocator=SimpleNamespace(cfg=SufficiencyConfig());run.robot=SimpleNamespace(set_positions=lambda *args:None)
    eligible=[o for o in observations if o.get('received_s',o['time_s'])<=start+T+1e-8]
    stamps=np.array([o.get('received_s',o['time_s']) for o in eligible]);frozen=None
    estimates={}
    def target(name,now):
        e=estimates.get(name)
        return e if e is not None and 0<=now-e.stamp<=.25 else None
    run.guide=SimpleNamespace(estimates=estimates,target=target)
    def refresh():
        nonlocal frozen
        # Never round up into a packet whose exposure/receive time is still
        # in the future, even by a floating-point-sized amount.
        index=int(np.searchsorted(stamps,run.time,side='right')-1)
        if index<0:raise ValueError('missing original received observation')
        if run.time>=onset:
            if fault=='exception':raise RuntimeError('injected sensor worker failure')
            if fault in ('missing','delayed'):
                if frozen is None:frozen=copy.deepcopy(estimates)
                estimates.clear();estimates.update(copy.deepcopy(frozen));return None
        record=eligible[index]['targets'].get(run.active)
        if record is None:raise ValueError('original active target missing')
        estimate=TargetEstimate(run.active,np.array(record['position']),np.array(record['covariance']),record['pixels'],record['stamp'],record.get('kind','magnetic_interface'))
        if run.time>=onset:
            if fault=='pixels':estimate.pixels=3
            elif fault=='covariance':estimate.covariance=np.eye(3)*.008**2
        estimates[run.active]=estimate
        return None
    refresh();run.sensor_service=SimpleNamespace(submit=lambda *args:None,poll=refresh)
    run.accept_packet=lambda packet:None;run.observe=lambda **kwargs:None
    reason=None
    # Suppress wall pacing only in this offline process. No timing claim.
    with patch('astronaut_delivery.rgbd_cooperative.time.sleep',lambda seconds:None):
        try:run._execute(curve)
        except RuntimeError as exc:reason=str(exc)
    q=np.array([s['q'] for s in run.steps]);delta=curve.end-curve.start;norm2=delta@delta
    if norm2<=1e-20:raise ValueError('nonmoving segment')
    progress=(q-curve.start)@delta/norm2
    residual=float(np.max(abs(q-curve.start-progress[:,None]*delta)))
    braking=[s for s in run.steps if s['phase']=='braking'];last=run.steps[-1]
    delay=braking[0]['time_s']-onset if braking else None
    deadline=(.25 if fault in ('missing','delayed') else 0.)+1/60
    timely=bool(delay is not None and -1e-8<=delay<=deadline+1e-8)
    rest=bool(max(np.linalg.norm(last['v']),np.linalg.norm(last['a']))<1e-6)
    line=bool(progress.min()>=-1e-8 and progress.max()<=1+1e-8 and residual<1e-8 and np.min(np.diff(progress))>=-1e-8)
    return dict(fault=fault,onset_fraction=fraction,phase=path.get('phase','approach'),visual_target=run.active,
        original_start_s=start,reason=reason,detected_after_onset_s=delay,detection_deadline_s=deadline,detection_within_bound=timely,
        rest_after_onset_s=run.time-onset,terminal_C2_rest=rest,within_original_monotone_line=line,
        maximum_joint_line_residual_rad=residual,terminal_progress=float(progress[-1]),
        inherited_geometry_valid=bool(path['environment_certificate']['allowed'] and path['robot_certificate'] and line),
        passed=bool(reason and 'smooth_brake' in reason and braking and rest and line and timely),
        reaction=reaction_metrics([s['wrench'] for s in run.steps],[s['dt_s'] for s in run.steps]))


def evaluate(run,scene):
    root=Path(run);audit=json.loads((root/'mission_audit.json').read_text())
    if json.loads((root/'result.json').read_text()).get('environment_authority')!='RGBD_ONLY':raise ValueError('pure RGB-D logs required')
    if not audit.get('scene_all_objects_delivered_audited'):raise ValueError('audited whole mission required for this replay protocol')
    paths=json.loads((root/'paths.json').read_text());observations=json.loads((root/'observations.json').read_text())
    selected={}
    for path in paths:
        phase=path.get('phase','approach')
        if phase in ('approach','lift','transport','place','retreat'):selected.setdefault((phase,path['visual_target_name']),path)
    stage=Usd.Stage.Open(str(scene));model=CombinedReaction(stage,*[TightBoundArm(stage,s) for s in ('left','right')])
    rows=[replay_case(p,observations,model,fault,fraction) for p in selected.values()
        for fraction in (.2,.5,.8) for fault in ('missing','pixels','covariance','exception')]
    return dict(scope='Offline recorded-feedback fault injection into unchanged streaming controller, not new RGB-D scene acquisition',
        source_run=str(root),selected_segments=len(selected),selection='First segment per (ordinary phase, measured visual target)',
        planned_cases=len(rows),passed_cases=sum(r['passed'] for r in rows),all_passed=all(r['passed'] for r in rows),cases=rows,
        caveats=['No rendered recovery or full-task completion under faults','Static original scene and ownership certificate required',
            'No wall-time performance claim; pacing disabled only in this offline test','Missing stream tests freshness expiry; no fabricated timestamps'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    with threadpool_limits(1):report=evaluate(a.run,a.scene)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
