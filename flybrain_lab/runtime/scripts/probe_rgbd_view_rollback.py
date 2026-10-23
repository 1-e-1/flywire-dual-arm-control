"""Saved-map camera return certificate, never actual execution or scene editing."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_rgbd_handover import saved_map
from astronaut_delivery.rgbd_transfer import RGBDTransfer
from astronaut_delivery.simultaneous import TightBoundArm,SimultaneousGuard
from astronaut_delivery.robot_only_geometry import RobotOnlyGuard
from astronaut_delivery.motion import Payload
from astronaut_delivery.vision import TargetEstimate
from astronaut_delivery.reaction import CombinedReaction
from astronaut_delivery.visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from classical_fusion.payload_contact_guard import SupportContactColumn


def restore(run):
    root=Path(run);r=json.loads((root/'result.json').read_text());cfg=json.loads((root/'run_configuration.json').read_text())
    paths=json.loads((root/'paths.json').read_text());last=paths[-1]
    if last.get('payload_contact_phase') not in ('lift','transport','place'):raise ValueError('Probe requires a saved attached-payload phase')
    t=RGBDTransfer.__new__(RGBDTransfer);t.q=np.asarray(r['final_joints']);t.time=r['simulation_time_s'];t.paths=paths
    t.steps=json.loads((root/'steps.json').read_text());t.observations=json.loads((root/'observations.json').read_text())
    t.active=last['visual_target_name'];t.guide=SimpleNamespace(estimates={k:TargetEstimate(k,np.asarray(v['position']),
        np.asarray(v['covariance']),v['pixels'],v['stamp'],v['kind']) for k,v in t.observations[-1]['targets'].items()})
    t.guide.target=lambda name,now:t.guide.estimates.get(name) if name in t.guide.estimates and 0<=now-t.guide.estimates[name].stamp<=.25 else None
    t.scan_source=(cfg['initial_scan'],cfg['scene_token'],cfg['calibration_id'],cfg['assume_unchanged_scene'])
    t.static_interval_fusion=cfg['static_interval_fusion']
    stage=Usd.Stage.Open(cfg['scene']);t.left,t.right=[TightBoundArm(stage,s) for s in ('left','right')]
    t.self_guard=SimultaneousGuard(RobotOnlyGuard(stage,t.left,t.right));t.mapper=saved_map(root);t.local_mapper=saved_map(root/'local_map')
    decisions=json.loads((root/'decisions.json').read_text())
    blocks=[d for d in decisions if d.get('status')=='ROBOT_COMPONENT_RAW_RGBD_REFINEMENT']
    if blocks:
        block=blocks[-1]
        t.robot_refinement=dict(mapper=saved_map(root/f'robot_refinement_{block["generation"]:02d}'),
            names=block['names'],generation=block['generation'])
    t.world_yaw_free=cfg.get('right_task_orientation')=='fixed_world_vertical_row_free_yaw'
    # Default restores the historical saved-run query scope; --broad-static
    # explicitly tests the revised policy when diagnosing older runs.
    if cfg.get('static_interval_scope','fine_only')=='fine_only':
        from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
        t.broad_snapshot=lambda:RobotQuerySnapshot(t.mapper)
    attachment=[e for e in r['transfer_events'] if e['event']=='measured_payload_attached'][-1]
    t.held=Payload(attachment['name'],np.asarray(attachment['local_points']),np.eye(4));t.source_tcp=np.asarray(attachment['tcp'])
    t.transfer_phase=last['payload_contact_phase'];t.decisions=[];t.planning_wall=[];t.task_metadata={}
    t.mode=cfg['mode'];t.recovery_count=0
    t.reaction=CombinedReaction(stage,t.left,t.right)
    t.allocator=VisualSetAllocator(t.left,t.reaction,SufficiencyConfig(objective='wrench',accept_feasible_iterate=True,cap_radius_m=.045,endpoint_reserve=.01))
    rotations=[p['right_task_rotation'] for p in paths if p['time_s']>=attachment['time_s'] and 'right_task_rotation' in p]
    t.transport_rotation=np.asarray(rotations[-1]) if rotations else t.source_tcp[:3,:3]
    if t.transfer_phase=='lift':
        c=last['contact_column'];column=SupportContactColumn.__new__(SupportContactColumn)
        column.equations=np.asarray(c['equations']);column.lower=c['lower'];column.upper=c['upper'];column.links=set(c['links'])
        t.source_column=column;t.initial_payload_bottom=column.lower+.009
        lift=next(p for p in reversed(paths) if p.get('phase')=='lift' and p['time_s']>=attachment['time_s'])
        t.saved_placement_goal=np.asarray(lift['target'])+np.array([0,0,lift['standoff_m']])
    if t.transfer_phase=='place':
        c=last['contact_column'];column=SupportContactColumn.__new__(SupportContactColumn)
        column.equations=np.asarray(c['equations']);column.lower=c['lower'];column.upper=c['upper'];column.links=set(c['links'])
        t.place_column=column;t.expected_placed_bottom=column.lower+.009
        placement=next(p for p in paths if p.get('phase')=='place' and p['time_s']>=attachment['time_s'])
        t.place_xy=np.asarray(placement['target'])[:2]
        t.saved_placement_goal=np.asarray(placement['target'])+np.array([0,0,placement['standoff_m']])
    requested=r.get('requested_transfer_goal',{})
    if requested.get('phase')==t.transfer_phase:t.saved_placement_goal=np.asarray(requested['position'],float)
    return t


def probe(run,place_step=False,broad_static=False):
    t=restore(run)
    if broad_static:
        from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot
        original=t.make_environment
        def changed(*args):
            env,_=original(*args);s=StaticIntervalSnapshot(t.mapper,assume_static_epoch=True)
            env.broad.snapshot=s;env.snapshot=s
            return env,s
        t.make_environment=changed
    if place_step:
        visual=t.guide.estimates[t.active]
        waypoint=TargetEstimate('measured_waypoint',t.saved_placement_goal,visual.covariance,visual.pixels,visual.stamp)
        motion,why=t.plan_step(waypoint,0.,step_m=.003)
        print(json.dumps(dict(scope='SAVED-MAP TRANSFER PROPOSAL ONLY; NOT EXECUTED',phase=t.transfer_phase,broad_static=broad_static,
            candidate=motion is not None,reason=why,decisions=t.decisions[-5:]),indent=2));return
    proposed=[];t.execute=lambda motion:proposed.append(dict(duration_s=motion.duration,start=motion.state(0)[0].tolist(),end=motion.state(motion.duration)[0].tolist()))
    allowed=t.recover_missing_task_view()
    print(json.dumps(dict(scope='SAVED-MAP CERTIFICATE ONLY; execution replaced by recording stub',allowed=allowed,
        proposals=proposed,decisions=t.decisions),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--place-step',action='store_true');p.add_argument('--broad-static',action='store_true');a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.place_step,a.broad_static)
