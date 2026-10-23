"""Authenticated saved-map diagnostic. Builds evidence; NEVER executes motion."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.probe_rgbd_view_rollback import restore
from astronaut_delivery.temporal_evidence_map import TemporalEvidenceTSDF
from astronaut_delivery.measured_task_memory import replay_changes
from astronaut_delivery.vision import TargetEstimate
from astronaut_delivery.source_freeze import freeze,unchanged
from astronaut_delivery.measured_surface import export_surface
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.rgbd_recording import replay_initial_scan


def probe(run,output,names):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    hashes=freeze(ROOT,folder,dict(run=run,names=names,scope='SAVED-MAP PROPOSAL; NO ACTUATION'))
    t=restore(run);hulls=t.right.hulls_world(t.q[1]);v=np.concatenate([hulls[n][0] for n in names])
    lo=v.min(0)-.04;hi=v.max(0)+.04
    if np.prod(np.ceil((hi-lo)/.002))>2_000_000:raise ValueError('refinement cell budget')
    cfg=PlanningMapConfig(workspace_min=tuple(lo),workspace_max=tuple(hi),voxel_size_m=.002,truncation_m=.006,
        surface_min_weight=1.,snapshot_max_age_s=1000000.)
    mapper=TemporalEvidenceTSDF(cfg,device='cuda:0');mapper.patch_radii=(4,6,8,10,12,16,24,32)
    scan,scene,calibration,static=t.scan_source
    evidence=replay_initial_scan(mapper,scan,expected_scene_token=scene,expected_calibration_id=calibration,assume_unchanged_scene=static)
    old=json.loads((Path(run)/'result.json').read_text());changes=replay_changes(mapper,old['transfer_events'])
    t.robot_refinement=dict(mapper=mapper,names=names,generation=1)
    last=t.paths[-1];goal=np.array(last['target'])+[0,0,last['standoff_m']]
    if last.get('phase')=='transport_height_adjustment':
        regular=next(p for p in reversed(t.paths) if p.get('phase')=='transport')
        goal[:2]=regular['target'][:2]  # diagnostic must advance toward destination, not stay at last escape endpoint
    target=t.guide.estimates[t.active]
    waypoint=TargetEstimate('measured_waypoint',goal,target.covariance,target.pixels,target.stamp)
    motion,reason=t.plan_step(waypoint,0.,step_m=.005)
    out=dict(scope='COMPLETE SAVED-MAP PROPOSAL ONLY; NOT EXECUTED OR LIVE-CLOCK CERTIFIED',candidate=motion is not None,
        reason=reason,goal=goal.tolist(),evidence=evidence,replayed_changes=changes,cells=int(np.prod(mapper.shape)),
        names=names,decisions=t.decisions[-4:],proposal=t.paths[-1] if motion is not None else None,
        source_unchanged=unchanged(ROOT,hashes))
    export_surface(mapper,folder)
    (folder/'probe.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('decisions','proposal')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--names',nargs='+',required=True);a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.output,a.names)
