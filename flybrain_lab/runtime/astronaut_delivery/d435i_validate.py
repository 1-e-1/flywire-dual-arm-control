"""Offline validation ONLY: known target CAD positions are used as metric labels.

This module is never imported by the observation or control policy.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd
from threadpoolctl import threadpool_limits
from .scene import ROOT
from .d435i import D435iCamera,load_profile
from .motion import Arm
from .vision import VisualGuide
from .run import DualArmKinematics

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'reports/astronaut_delivery_d435i_verified');args=p.parse_args()
    folder=args.run;result=json.loads((folder/'result.json').read_text());records=json.loads((folder/'trajectory.json').read_text())
    cfg=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
    # Evaluation labels only; control obtained these positions from RGB-D.
    truth={p['name']:np.array(p['position'])+[0,0,.055] for p in cfg['parts']}
    poses=[r for r in records if r['phase'] in ('initial_view/settled','mapping_second_view/settled')]
    if len(poses)!=2:raise RuntimeError('Expected two independently observed views')
    stage=Usd.Stage.Open(result['scene_path']);stage.SetEditTarget(stage.GetSessionLayer());left=Arm(stage,'left');robot=DualArmKinematics(stage)
    rows=[];saved=False
    for multiplier in (1.,3.):
        profile=copy.deepcopy(load_profile());profile['simulation_noise']['floor_sigma_m']*=multiplier;profile['simulation_noise']['disparity_sigma_px']*=multiplier
        sensor=D435iCamera(profile=profile)
        for view,row in enumerate(poses):
            q=np.array(row['q']);robot.set_positions(*q);camera_world=left.camera(q[0])
            errors=[];ratios=[];durations=[];counts=[]
            for seed in (17,41,20260906):
                guide=VisualGuide({s['name']:s['color'] for s in cfg['parts']},sensor_profile=profile)
                start=time.perf_counter();obs=sensor.capture(stage,camera_world,0.,seed=seed);durations.append(time.perf_counter()-start)
                guide.update(obs,integrate_map=False)
                error={name:float(np.linalg.norm(v.position-truth[name])) for name,v in guide.estimates.items()}
                errors.append(error);counts.append(len(error));ratios.append(sensor.last_quality)
                if not saved and view==1:
                    Image.fromarray((sensor.last_native_color*255).astype(np.uint8)).save(folder/'d435i_native_rgb.png')
                    Image.fromarray((obs.rgb*255).astype(np.uint8)).save(folder/'d435i_rgb_on_depth.png')
                    np.savez_compressed(folder/'d435i_initial_observation.npz',depth_m=obs.depth,rgb_on_depth=obs.rgb,
                                        color_valid=obs.color_valid,robot_mask=obs.robot_mask,K_depth=obs.K,camera_world_gl=obs.camera_world)
                    saved=True
            rows.append(dict(noise_multiplier=multiplier,view=view,seeds=[17,41,20260906],target_center_error_m=errors,
                             detected_counts=counts,quality=ratios,capture_wall_s=durations,
                             distance_to_evaluation_targets_m={n:float(np.linalg.norm(v-camera_world[:3,3])) for n,v in truth.items()}))
    events=json.loads((folder/'events.json').read_text());gaps=[e['capture_gap_m'] for e in events if e['event']=='attached']
    errors=[e for row in rows for sample in row['target_center_error_m'] for e in sample.values()]
    nominal=[row for row in rows if row['noise_multiplier']==1]
    passed=(result['status']=='PASS' and result['continuous_audit'] and len(result['completed'])==3
            and len(gaps)==3 and max(gaps)<=cfg['magnet_capture_distance_m']
            and all(c==3 for row in nominal for c in row['detected_counts'])
            and max(e for row in nominal for sample in row['target_center_error_m'] for e in sample.values())<.008)
    report=dict(status='PASS' if passed else 'FAIL',scope='One simulated workcell, two selected views, three fixed sensor seeds. NOT hardware validation or a success-rate study.',
                completed=result['completed'],capture_gaps_m=gaps,static_view_noise_ablation=rows,
                maximum_cap_center_error_all_stress_samples_m=max(errors),
                stress_capture_tolerance_status='WITHIN_TOLERANCE' if max(errors)<cfg['magnet_capture_distance_m'] else 'EXCEEDS_8_MM_CAPTURE_TOLERANCE_DO_NOT_ASSUME_ROBUST_CAPTURE',
                calibration_status='NOMINAL_SIMULATION_ONLY',depth_noise_model='Engineering assumption; not measured D435i accuracy',
                limitations=['12 mm prototype TSDF, not a verified millimetre planning map','Face/material ray renderer; no real stereo matching or lighting',
                             'Known coloured caps, not general unknown-object detection','Right-arm safety still uses known CAD, not a perception-only safety proof',
                             'No hand-eye calibration or exposure-time robot synchronization validated'])
    (folder/'d435i_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    with threadpool_limits(1):main()
