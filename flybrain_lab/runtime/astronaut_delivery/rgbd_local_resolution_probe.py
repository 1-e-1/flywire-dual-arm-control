"""Offline raw-RGBD local-resolution diagnostic; never an executable shortcut."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from classical_fusion.planning_map import PlanningMapConfig
from classical_fusion.rgbd_recording import replay_initial_scan
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from classical_fusion.rgbd_guard import RGBDChordEnvironmentGuard
from classical_fusion.measured_contact import MeasuredContactPermit
from classical_fusion.measured_object import propose_envelope
from .temporal_evidence_map import TemporalEvidenceTSDF
from .simultaneous import TightBoundArm
from .measured_surface import export_surface


class FixedHull:
    def __init__(self,vertices,normals):
        self.vertices,self.normals=vertices,normals;self.motion_radius=.2
        self.center=np.mean(vertices,axis=0);self.radii=np.array([np.linalg.norm(vertices-self.center,axis=1).max()])
        self.sphere_names=['magnet']
    def sphere_positions(self,q):return self.center[None]
    def absolute_motion_bound(self,name,delta):return 0.
    def hulls_world(self,q):return {'magnet':(self.vertices,self.normals)}


def probe(run,scan,scene,output):
    root=Path(run);destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    result=json.loads((root/'result.json').read_text());q=np.array(result['final_joints'])
    target=np.array(json.loads((root/'observations.json').read_text())[-1]['targets']['bracket']['position'])
    cfg=PlanningMapConfig(workspace_min=tuple(target+[-.13,-.13,-.04]),workspace_max=tuple(target+[.13,.13,.20]),
        voxel_size_m=.002,truncation_m=.006,surface_min_weight=1.,snapshot_max_age_s=300.)
    mapper=TemporalEvidenceTSDF(cfg,device='cuda:0');mapper.patch_radii=(4,6,8,10,12,16,24,32)
    manifest=json.loads(Path(scan).read_text())
    replay_initial_scan(mapper,scan,expected_scene_token=manifest['scene_token'],expected_calibration_id=manifest['calibration_id'],assume_unchanged_scene=True)
    snapshot=RobotQuerySnapshot(mapper)
    stage=Usd.Stage.Open(scene);arm=TightBoundArm(stage,'right')
    envelope=propose_envelope(np.load(root/'measured_surface.npz')['vertices'],target)
    permit=MeasuredContactPermit.from_envelope(envelope,target,np.load(root/'measured_surface.npz')['vertices']);rows=[]
    for gap in (.04,.02,.008,.005):
        goal=arm.solve(target+[0,0,gap],[0,0,-1],seed=q[1],tries=4)
        if goal is None:rows.append(dict(gap=gap,IK=False));continue
        vertices,normals=arm.hulls_world(goal)['magnet'];fixed=FixedHull(vertices,normals)
        guard=RGBDChordEnvironmentGuard(fixed,fixed,snapshot)
        strict=guard.segment(np.zeros(16),np.zeros(16))
        guard.authorized_contact_cells=lambda name,centers:permit.cells(name,centers,snapshot.spacing)
        contact=guard.segment(np.zeros(16),np.zeros(16))
        rows.append(dict(gap=gap,strict=strict.allowed,contact=contact.allowed,reason=contact.reason,rejection=guard.last_rejection))
    out=dict(scope='Local 2 mm raw-RGBD replay and fixed-tool queries only; no motion, no live use of saved grids',
        free_fraction=snapshot.free_fraction,independent_views=len(mapper.evidence_poses),queries=rows)
    export_surface(mapper,destination);(destination/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scan',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.scan,a.scene,a.output)
