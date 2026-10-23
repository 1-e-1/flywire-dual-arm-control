"""Saved measured-map transport diagnostic; no execution or CAD obstacles.

All proposals retain the actual held envelope and complete robot/environment
segment guards. A bounded world-Z yaw fallback is now also implemented online;
this saved-map diagnostic itself never executes a proposal.
"""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_rgbd_handover import saved_map
from pxr import Usd
from astronaut_delivery.rgbd_transfer import RGBDTransfer
from astronaut_delivery.simultaneous import TightBoundArm,SimultaneousGuard
from astronaut_delivery.robot_only_geometry import RobotOnlyGuard
from astronaut_delivery.motion import Payload
from astronaut_delivery.vision import TargetEstimate
from astronaut_delivery.evidence_ik import residual_from_rejections


def probe(run,scene):
    root=Path(run);result=json.loads((root/'result.json').read_text())
    cfg=json.loads((root/'run_configuration.json').read_text())
    if 'transfer/transport/' not in str(result.get('error','')):
        raise ValueError('This probe requires a stopped transport failure')
    observations=json.loads((root/'observations.json').read_text())
    paths=json.loads((root/'paths.json').read_text())
    attachment=[e for e in result['transfer_events'] if e['event']=='measured_payload_attached'][-1]
    task=RGBDTransfer.__new__(RGBDTransfer);task.q=np.asarray(result['final_joints'])
    task.scan_source=(cfg['initial_scan'],cfg['scene_token'],cfg['calibration_id'],cfg['assume_unchanged_scene'])
    task.static_interval_fusion=cfg['static_interval_fusion']
    task.time=result['simulation_time_s'];task.transfer_phase='transport'
    task.mapper=saved_map(root);task.local_mapper=saved_map(root/'local_map')
    stage=Usd.Stage.Open(scene)
    task.left,task.right=[TightBoundArm(stage,s) for s in ('left','right')]
    task.self_guard=SimultaneousGuard(RobotOnlyGuard(stage,task.left,task.right))
    task.held=Payload(attachment['name'],np.asarray(attachment['local_points']),np.eye(4))
    task.source_tcp=np.asarray(attachment['tcp'])
    target_name=next(p['visual_target_name'] for p in reversed(paths)
                     if p.get('payload_contact_phase')=='transport')
    e=observations[-1]['targets'][target_name]
    target=TargetEstimate(target_name,np.asarray(e['position']),np.asarray(e['covariance']),e['pixels'],e['stamp'],e['kind'])
    tcp=task.right.tcp(task.q[1]);q0=task.q.ravel()
    direction=np.r_[target.position[:2],tcp[2,3]]-tcp[:3,3]
    env,_=task.make_environment(target,0.)
    print(json.dumps(dict(scope='Saved measured-only transport proposal; NOT executed',
        target_measurement_age_s=task.time-target.stamp,stationary_allowed=env.segment(q0,q0).allowed,
        current_tcp=tcp[:3,3].tolist(),destination_xy=target.position[:2].tolist())),flush=True)
    for step in (.02,.01,.005):
        goal=tcp[:3,3]+direction*min(1.,step/max(np.linalg.norm(direction),1e-9))
        for yaw in (0,15,-15,30,-30):
            rotation=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@task.source_tcp[:3,:3]
            rejections=[];began=time.perf_counter()
            def valid(right):
                pair=np.r_[task.q[0],right]
                if not task.self_guard.segment(q0,pair,held=task.held):return False
                certificate=env.segment(q0,pair)
                if not certificate.allowed:rejections.append(env.last_rejection)
                return certificate.allowed
            candidate=task.right.solve(goal,[0,0,-1],seed=task.q[1],rotation=rotation,
                max_delta=.30,tries=8,valid=valid,rng=np.random.default_rng(32001))
            if candidate is None:
                residual=residual_from_rejections(task.right,rejections,start=task.q[1])
                if residual is not None:
                    candidate=task.right.solve(goal,[0,0,-1],seed=task.q[1],rotation=rotation,max_delta=.30,
                        tries=4,valid=valid,extra_residual=residual,rng=np.random.default_rng(32002))
            print(json.dumps(dict(step_m=step,world_yaw_deg=yaw,candidate=candidate is not None,
                joints=None if candidate is None else candidate.tolist(),
                last_rejection=rejections[-1] if rejections else None,wall_s=time.perf_counter()-began)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.scene)
