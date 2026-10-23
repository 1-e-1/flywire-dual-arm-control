"""Diagnose pixel-support versus voxel/error-budget mismatch using real rays."""
import argparse,json,gc
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from pxr import Usd,Gf
from threadpoolctl import threadpool_limits
from .online_scene import ONLINE_SCENE
from .run import DualArmKinematics
from .d435i import D435iCamera
from .vision import VisualGuide
from classical_fusion.planning_map import PlanningTSDF,PlanningMapConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    source=json.loads((Path(args.run)/'telemetry/observations.json').read_text())
    rows=[min(source,key=lambda r:abs(r['exposure_s']-t)) for t in (13.65,20.,27.,34.)]
    stage=Usd.Stage.Open(str(ONLINE_SCENE));stage.SetEditTarget(stage.GetSessionLayer());robot=DualArmKinematics(stage)
    camera=D435iCamera();observations=[]
    for i,r in enumerate(rows):
        robot.set_positions(*np.array(r['q']).reshape(2,8));observations.append(camera.capture(stage,np.array(r['camera_world']),r['exposure_s'],seed=118+i))
    reports=[]
    with threadpool_limits(1):
        for voxel in (.012,.004):
            for radius in (5,8,12,16):
                guide=VisualGuide({},sensor_profile=camera.profile)
                guide.mapper=PlanningTSDF(replace(guide.mapper.cfg,voxel_size_m=voxel,truncation_m=voxel*3,free_patch_radius_px=radius),device='cuda:0')
                for obs in observations:guide.update(obs)
                votes=guide.mapper.free_views
                reports.append(dict(voxel_m=voxel,patch_radius_px=radius,one_view_free_cells=int((votes>=1).sum()),
                                    two_view_free_cells=int(((votes>=2)&~guide.mapper.surface_veto).sum()),total_cells=votes.numel()))
                print(json.dumps(reports[-1]),flush=True);del guide;gc.collect();torch.cuda.empty_cache()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(scope='Four full-frame USD ray observations before first attachment; diagnostic, not a mission or a calibration.',rows=reports),indent=2),encoding='utf-8')


if __name__=='__main__':main()
