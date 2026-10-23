"""Native Isaac Sim load/render/zero-gravity stability check, not an RL task."""
import argparse
import os
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--headless',action='store_true')
args=parser.parse_args()
check_dir=Path(__file__).resolve().parents[1]/'logs'/'isaacsim_asset_check_cache'
check_dir.mkdir(parents=True,exist_ok=True)
os.environ['OPTIX_CACHE_PATH']=str(check_dir/'optix')
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':960,'height':720,'renderer':'RaytracedLighting','anti_aliasing':2,'extra_args':[f'--/app/tokens/cache={check_dir.as_posix()}',f'--/app/tokens/data={(check_dir/"data").as_posix()}',f'--/app/tokens/logs={(check_dir/"logs").as_posix()}']})
import json
import numpy as np
import omni.usd
import omni.timeline
import omni.replicator.core as rep
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from build_astronaut_dual_arm import USD, OUT, world, matrix, transform, digest
from astronaut_kinematics import DualArmKinematics

try:
    (OUT/'isaacsim_validation.json').write_text(json.dumps(dict(status='RUNNING')),encoding='utf-8')
    omni.usd.get_context().open_stage(str(USD))
    for _ in range(20):
        app.update()
    stage=omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    robot=DualArmKinematics(stage)
    tracked=['/World/Astronaut/Body','/World/C_structure','/World/LeftArm/base/a1','/World/RightArm/base/a1','/World/RightMagnet']
    initial={p:np.array(world(stage.GetPrimAtPath(p))) for p in tracked}
    timeline=omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
    drift={p:float(np.max(np.abs(np.array(world(stage.GetPrimAtPath(p)))-initial[p]))) for p in tracked}
    images={}
    for name,path in [('overview','/World/OverviewCamera'),('left_camera','/World/LeftArm/b8/Camera'),('left_camera_check','/World/LeftArm/b8/Camera')]:
        if name=='left_camera_check':
            # Session layer ONLY: this target is never saved into the main USD.
            cam_world=world(stage.GetPrimAtPath(path))
            for x in range(5):
                for y in range(5):
                    cube=UsdGeom.Cube.Define(stage,f'/World/CameraTestTarget/tile_{x}_{y}')
                    cube.CreateSizeAttr(1)
                    transform(cube.GetPrim(),matrix(np.diag([.07,.07,.01]),((x-2)*.07,(y-2)*.07,-.7))*cam_world)
                    cube.CreateDisplayColorAttr([Gf.Vec3f(.02,.1,.7) if (x+y)%2 else Gf.Vec3f(.85,.15,.03)])
        product=rep.create.render_product(path,(960,720))
        annotator=rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
        for _ in range(12):
            rep.orchestrator.step(rt_subframes=2)
        pixels=np.asarray(annotator.get_data())
        assert pixels.shape[:2]==(720,960),(name,pixels.shape)
        target=OUT/f'isaacsim_{name}.png'
        Image.fromarray(pixels).save(target)
        images[name]=dict(path=str(target),shape=list(pixels.shape),std=float(pixels[:,:,:3].std()))
        images[name]['spatial_std']=float(pixels[:,:,:3].astype(float).mean(2).std())
        if name=='left_camera_check':
            assert images[name]['spatial_std']>10,'Camera did not see the test target'
        annotator.detach(product)
        product.destroy()
    timeline.stop()
    assert max(drift.values())<1e-6,drift
    result=dict(status='PASS',steps=120,max_pose_matrix_drift=drift,images=images,note='Runtime static stability and render test. left_camera is expected empty; left_camera_check has a session-only target absent from the final USD. NOT collision-free motion certification.')
    result['usd_sha256']=digest(USD)
    result['astronaut_pose_sha256']=digest(OUT/'astronaut'/'astronaut_work_pose.usd')
    (OUT/'isaacsim_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ASTRONAUT_RUNTIME_RESULT '+json.dumps(result),flush=True)
except Exception as exc:
    (OUT/'isaacsim_validation.json').write_text(json.dumps(dict(status='FAIL',error=str(exc))),encoding='utf-8')
    raise
finally:
    # All RGB reads/writes above are synchronous and products are detached.
    # Do not wait for an idle replicator workflow. Kit stage shutdown may still
    # hang after an internal SharedMutex assertion; that is an engine limitation,
    # not evidence that the saved USD or the completed render checks failed.
    app.close(wait_for_replicator=False)
