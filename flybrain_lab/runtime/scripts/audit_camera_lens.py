"""Native DirectRLEnv camera calibration audit; a ground plane, static arm.

Compare requested USD focal scale with independently measured plane residual.
The camera's own K readback is insufficient to validate the rendered image.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--output',default=str(ROOT/'classical_fusion/outputs/camera_lens_audit.json'))
parser.add_argument('--resolution',type=int,default=256)
parser.add_argument('--principal_point_audit',action='store_true')
args=parser.parse_args()
launcher=AppLauncher(args)
app=launcher.app

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
import isaaclab.sim as sim_utils
from codex_zbot.tasks.active_vision_env import ActiveVisionEnvCfg,VISIBLE_INITIAL_POSES
from codex_zbot.tasks.scene import pinhole_intrinsics,configure_camera_intrinsics
from codex_zbot.perception.pointcloud import backproject_depth_world

class AuditEnv(DirectRLEnv):
    def _setup_scene(self):
        self.robot=Articulation(self.cfg.robot)
        self.camera=Camera(self.cfg.camera)
        # A procedural plane avoids downloading Isaac's default floor USD.
        # Its upper surface is exactly z=0, our calibration reference.
        floor=sim_utils.CuboidCfg(size=(4.,4.,.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.5,.5,.5)))
        floor.func('/World/ground',floor,translation=(0.,0.,-.01))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot']=self.robot
        self.scene.sensors['camera']=self.camera
        light=sim_utils.DomeLightCfg(intensity=1800.)
        light.func('/World/Light',light)
    def __init__(self,cfg):
        super().__init__(cfg)
        self.joint_ids,_=self.robot.find_joints('joint[1-8]')
        self.q=self.robot.data.default_joint_pos.clone()
        self.q[:,self.joint_ids]=torch.tensor(VISIBLE_INITIAL_POSES[0],device=self.device)
    def _pre_physics_step(self,actions):
        pass
    def _apply_action(self):
        self.robot.write_joint_state_to_sim(self.q,torch.zeros_like(self.q))
        self.robot.set_joint_position_target(self.q)
    def _get_observations(self):
        return {'policy':torch.zeros(self.num_envs,1,device=self.device)}
    def _get_rewards(self):
        return torch.zeros(self.num_envs,device=self.device)
    def _get_dones(self):
        zero=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        return zero,zero
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)

def main():
    cfg=ActiveVisionEnvCfg()
    cfg.scene.num_envs=1
    cfg.sim.device=args.device
    cfg.objects=()
    cfg.camera.width=cfg.camera.height=args.resolution
    cfg.observation_space=1; cfg.state_space=0
    env=AuditEnv(cfg)
    env.reset()
    action=torch.zeros(1,8,device=env.device)
    requested=pinhole_intrinsics(args.resolution,args.resolution,110,device=env.device)[None]
    records=[]
    if args.principal_point_audit:
        configure_camera_intrinsics(env.camera,110.)
        for pose_index,joints in enumerate(VISIBLE_INITIAL_POSES):
            env.q[:,env.joint_ids]=torch.tensor(joints,device=env.device)
            for _ in range(16):
                env.step(action)
            depth=env.camera.data.output['distance_to_image_plane'].clone()
            pose=torch.cat([env.camera.data.pos_w-env.scene.env_origins,env.camera.data.quat_w_world],1)
            k=env.camera.data.intrinsic_matrices.clone()
            for offset in (0.,.5):
                matrix=k.clone(); matrix[:,0,2]-=offset; matrix[:,1,2]-=offset
                points,valid=backproject_depth_world(depth,matrix,pose,2.)
                roi=valid&(points[...,0]>.45)&(points[...,0]<1.05)&(points[...,1].abs()<.65)
                z=points[...,2][roi]
                if len(z)<100:
                    raise RuntimeError('not enough floor points for principal-point audit')
                result={'pose_index':pose_index,'pixel_center_offset':offset,'intrinsics':matrix[0].cpu().tolist(),
                        'samples':len(z),'ground_mae_m':float(z.abs().mean()),
                        'ground_p95_m':float(torch.quantile(z.abs(),.95))}
                records.append(result); print('[CENTER-AUDIT]',json.dumps(result),flush=True)
        output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(records,indent=2),encoding='utf-8')
        env.close()
        return
    for focal in [None,24.0]:
        if focal is None:
            env.camera.set_intrinsic_matrices(requested,focal_length=None)
        else:
            # Exercise the shared collection setup in future regression runs.
            configure_camera_intrinsics(env.camera,110.)
        for _ in range(16):
            env.step(action)
        depth=env.camera.data.output['distance_to_image_plane'].clone()
        pose=torch.cat([env.camera.data.pos_w-env.scene.env_origins,env.camera.data.quat_w_world],1)
        readback=env.camera.data.intrinsic_matrices.clone()
        points,valid=backproject_depth_world(depth,readback,pose,2.)
        mask=valid & (points[...,0]>.45) & (points[...,0]<1.05) & (points[...,1].abs()<.65)
        # Ground-only scene: target geometry z=0; robot excluded by the work ROI.
        z=points[...,2][mask]
        if len(z)<100:
            raise RuntimeError('not enough ground samples for camera audit')
        result={'usd_focal_length':focal,'readback_K':readback[0].cpu().tolist(),
                'samples':len(z),'ground_mean_abs_z_m':float(z.abs().mean()),
                'ground_median_z_m':float(z.median())}
        records.append(result)
        print('[LENS-AUDIT]',json.dumps(result),flush=True)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(records,indent=2),encoding='utf-8')
    env.close()

try:
    main()
except Exception as exc:
    # SimulationApp.close may terminate Python before an uncaught exception
    # is printed. Preserve failure evidence rather than reporting exit=0 as
    # a successful camera calibration.
    traceback.print_exc()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'status':'failed','error':repr(exc)},indent=2),encoding='utf-8')
    raise
finally:
    app.close()
