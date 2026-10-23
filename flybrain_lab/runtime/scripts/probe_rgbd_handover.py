"""Non-mutating saved-RGB-D handover diagnostic. No simulated environment query."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
from itertools import product
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from astronaut_delivery.rgbd_transfer import RGBDTransfer
from astronaut_delivery.simultaneous import TightBoundArm,SimultaneousGuard
from astronaut_delivery.robot_only_geometry import RobotOnlyGuard
from astronaut_delivery.visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from astronaut_delivery.reaction import CombinedReaction
from astronaut_delivery.motion import Payload
from astronaut_delivery.vision import TargetEstimate
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.payload_contact_guard import SupportContactColumn


def saved_map(folder):
    meta=json.loads((folder/'measured_surface.json').read_text())
    with np.load(folder/'planning_map_state.npz') as z:
        cfg=PlanningMapConfig(**dict(meta['config'],snapshot_max_age_s=1e6))
        with np.load(folder/'measured_surface.npz') as mesh:
            v,f=torch.from_numpy(mesh['vertices']),torch.from_numpy(mesh['triangles'])
        return SimpleNamespace(_lock=RLock(),cfg=cfg,revision=0,latest_observation_time=time.monotonic(),shape=z['distance'].shape,mesh=lambda:(v,f),
            **{k:torch.from_numpy(z[k]) for k in ('distance','weight','free_views','surface_veto','lower','spacing')})


def probe(run,scene,acquire_target=None,quality_target=None,approach_target=None):
    root=Path(run);r=json.loads((root/'result.json').read_text());obs=json.loads((root/'observations.json').read_text())
    cfg=json.loads((root/'run_configuration.json').read_text())
    attachments=[e for e in r['transfer_events'] if e['event']=='measured_payload_attached']
    e=attachments[-1] if attachments else None
    if e is None and acquire_target is None and approach_target is None:raise ValueError('this diagnostic requires an attachment or an approach target')
    task=RGBDTransfer.__new__(RGBDTransfer);task.q=np.array(r['final_joints']);task.time=r['simulation_time_s']
    task.scan_source=(cfg['initial_scan'],cfg['scene_token'],cfg['calibration_id'],cfg['assume_unchanged_scene'])
    task.mapper=saved_map(root);task.local_mapper=saved_map(root/'local_map') if (root/'local_map').is_dir() else None
    stage=Usd.Stage.Open(scene);task.left,task.right=[TightBoundArm(stage,s) for s in ('left','right')];task.arms=[task.left,task.right]
    task.self_guard=SimultaneousGuard(RobotOnlyGuard(stage,*task.arms));task.reaction=CombinedReaction(stage,*task.arms)
    task.allocator=VisualSetAllocator(task.left,task.reaction,SufficiencyConfig(objective='wrench',accept_feasible_iterate=True))
    task.held=Payload(e['name'],np.array(e['local_points']),np.eye(4)) if e else None
    task.source_tcp=np.array(e['tcp']) if e else task.right.tcp(task.q[1])
    estimates={k:TargetEstimate(k,np.array(v['position']),np.array(v.get('covariance',np.eye(3)*.002**2)),v['pixels'],v['stamp'],kind=v['kind']) for k,v in obs[-1]['targets'].items()}
    task.guide=SimpleNamespace(estimates=estimates,target=lambda n,t:estimates[n] if n in estimates and 0<=t-estimates[n].stamp<=.25 else None)
    task.active=e['name'] if e else acquire_target or approach_target;task.transfer_phase='lift' if e else None;task.decisions=[];task.paths=[];task.recovery_count=0
    task.steps=[];task.contact_permission=True
    with np.load(root/'raw'/obs[-1]['file']) as z:task.camera=SimpleNamespace(K=z['K'])
    task.contact_permits=[];task.static_interval_fusion=True
    if approach_target is not None:
        task.held=None;task.transfer_phase=None;task.active=approach_target;task.mode='set'
        with np.load(root/'raw'/obs[-1]['file']) as z:task.camera=SimpleNamespace(K=z['K'])
        task.contact_permits=[];task.static_interval_fusion=True
        target=estimates[approach_target]
        for step in (.02,.01,.005):
            began=time.perf_counter();motion,reason=task.plan_step(target,.007,step_m=step)
            print(json.dumps(dict(step_m=step,candidate=motion is not None,reason=reason,wall_s=time.perf_counter()-began,
                scope='Saved measured-map proposal only; NOT executed')),flush=True)
            if motion is not None:break
        print(json.dumps(task.decisions[-2:],indent=2),flush=True);return
    if acquire_target is not None:
        task.held=None;task.transfer_phase=None;task.active=acquire_target;target=estimates[acquire_target]
        print('current visual features [u,v,range,elevation]',task.allocator.features(task.q[0],target.position)[0].tolist(),flush=True)
        env,_=task.make_environment(target,.10)
        print('stationary acquisition',env.segment(task.q.ravel(),task.q.ravel()),env.last_rejection,flush=True)
        gap=float(task.right.tcp(task.q[1])[2,3]-target.position[2])
        contact_env,_=task.make_environment(target,gap)
        print('stationary with actual standoff',gap,contact_env.segment(task.q.ravel(),task.q.ravel()),contact_env.last_rejection,flush=True)
        motion=task.plan_visual_acquisition(target)
        print(json.dumps(dict(scope='Saved-map diagnostic; NOT executed',candidate=motion is not None,decisions=task.decisions),indent=2),flush=True)
        return
    with np.load(root/e.get('envelope_file','captured_measured_envelope.npz')) as z:vertices=z['vertices']
    task.initial_payload_bottom=float(vertices[:,2].min())
    target=estimates[task.active]
    footprint=(vertices[:,None,:2]+np.array(list(product([-.009,.009],repeat=2)))[None]).reshape(-1,2)
    task.source_column=SupportContactColumn(footprint,task.initial_payload_bottom-.009,target.position[2]+.04,['payload','magnet'])
    if quality_target is not None:
        task.active=quality_target;task.recovery_count=0
        task.allocator.cfg=SufficiencyConfig(objective='wrench',accept_feasible_iterate=True,cap_radius_m=.045,endpoint_reserve=.01)
        try:task.allocation_context(estimates[quality_target]);reason='diagnostic_quality_request'
        except ValueError as exc:reason=str(exc)
        print('operational recovery reason',reason,flush=True)
        motion,witness=task.plan_operational_recovery(estimates[quality_target],reason)
        print(json.dumps(dict(scope='Saved-map quality recovery diagnostic; NOT executed; fresh new target remains required',
            candidate=motion is not None,witness=witness,decisions=task.decisions),indent=2),flush=True)
        return
    # Covariance is a stated diagnostic approximation; raw saved estimates do
    # not contain it. No diagnostic candidate is executed or called certified.
    places=[v for v in estimates.values() if v.kind=='placement_marking'];place=min(places,key=lambda p:np.linalg.norm(p.position-target.position))
    env,_=task.make_environment(target,0)
    print('stationary',env.segment(task.q.ravel(),task.q.ravel()),env.last_rejection,flush=True)
    motion=task.plan_observation_recovery(target,look_at=(target.position+place.position)*.5)
    print(json.dumps(dict(scope='SAVED-MAP DIAGNOSTIC, approximate covariance 2mm; NOT executed',candidate=motion is not None,decisions=task.decisions),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--acquire-target');p.add_argument('--quality-target');p.add_argument('--approach-target');a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.scene,a.acquire_target,a.quality_target,a.approach_target)
