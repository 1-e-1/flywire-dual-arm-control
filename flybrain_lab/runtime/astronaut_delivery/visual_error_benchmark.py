"""Independent scene-level visual error characterization using audit-only truth."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from pxr import Usd
from threadpoolctl import threadpool_limits
from .randomized_benchmark import draw,make_scene
from .d435i import D435iCamera,load_profile
from .run import DualArmKinematics
from .vision import VisualGuide
from .motion import world


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=20)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path('reports/continuous_visual_nominal_01/telemetry/observations.json').read_text())
    poses=[min(source,key=lambda r:abs(r['exposure_s']-t)) for t in (13.65,20.,27.,34.)]
    cameras={}
    for mode,factor,drop in [('nominal',1,.005),('noise2',2,.005),('noise4',4,.005),('dropout20',1,.2)]:
        profile=load_profile();noise=profile['simulation_noise']
        noise['floor_sigma_m']*=factor;noise['disparity_sigma_px']*=factor;noise['dropout_probability']=drop
        cameras[mode]=D435iCamera(profile=profile)
    records=[]
    with threadpool_limits(1):
        for seed in range(3000,3000+args.count):
            cfg=draw(seed,'combined');path=make_scene(out/f'scene_{seed}',cfg)
            stage=Usd.Stage.Open(str(path));stage.SetEditTarget(stage.GetSessionLayer());robot=DualArmKinematics(stage)
            for index,pose in enumerate(poses):
                robot.set_positions(*np.array(pose['q']).reshape(2,8))
                for mode,camera in cameras.items():
                    obs=camera.capture(stage,np.array(pose['camera_world']),float(index),seed=seed*10+index)
                    guide=VisualGuide({p['name']:p['color'] for p in cfg['parts']},sensor_profile=camera.profile)
                    guide.fit_circular_cap=True;guide.update(obs,integrate_map=False)
                    for spec in cfg['parts']:
                        name=spec['name'];estimate=guide.target(name,float(index));row=dict(seed=seed,view=index,mode=mode,part=name,detected=estimate is not None)
                        if estimate is not None:
                            prim=stage.GetPrimAtPath('/World/Parts/'+name)
                            truth=(np.array(world(prim)).T@np.r_[prim.GetAttribute('magnet:contactLocal').Get(),1])[:3]
                            error=estimate.position-truth;sigma=float(np.sqrt(np.linalg.eigvalsh(estimate.covariance).max()))
                            row.update(error_m=float(np.linalg.norm(error)),sigma_max_m=sigma,pixels=estimate.pixels,
                                       inside_engineering_3sigma=bool(np.linalg.norm(error)<=3*sigma),
                                       estimate=estimate.position.tolist(),truth_audit_only=truth.tolist())
                        records.append(row)
            print(json.dumps(dict(seed=seed,observations=len(records))),flush=True)
    summary={}
    for mode in cameras:
        attempted=[r for r in records if r['mode']==mode];accepted=[r for r in attempted if r['detected']]
        errors=np.array([r['error_m'] for r in accepted]);summary[mode]=dict(attempted_targets=len(attempted),detected_targets=len(accepted),
            conditional_error_m=dict(mean=float(errors.mean()),p95=float(np.quantile(errors,.95)),max=float(errors.max())) if len(errors) else None,
            engineering_3sigma_coverage=float(np.mean([r['inside_engineering_3sigma'] for r in accepted])) if accepted else None)
    result=dict(scope='Twenty independent scene draws and four recorded pre-grasp camera poses; calibrated nominal ray geometry, known colored horizontal docking interfaces. Repeated views/targets are clustered, not independent trials.',
                seeds=list(range(3000,3000+args.count)),summary=summary,records=records,
                limits='Detection failure retained; errors conditional on detected targets. Noise model is synthetic. Not physical calibration, blind-object recognition, active-policy success, or a probability certificate.')
    (out/'visual_error_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
