"""IsaacLab integration smoke test; random checkpoints are not research results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--steps", type=int, default=3)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import torch

from codex_zbot.tasks.active_vision_env import ActiveVisionEnv, ActiveVisionEnvCfg
from codex_zbot.perception.pointcloud import backproject_depth_world
from codex_zbot.tasks.scene import camera_metric_intrinsics


def main() -> None:
    cfg = ActiveVisionEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args_cli.device
    cfg.algorithm_config_path = str(ROOT / "configs" / "env_smoke.json")
    cfg.cfm_checkpoint = str(ROOT / "checkpoints" / "env_smoke" / "cfm.pt")
    cfg.belief_checkpoint = str(ROOT / "checkpoints" / "env_smoke" / "belief.pt")
    cfg.gt_points = 128
    env = ActiveVisionEnv(cfg)
    observation, _ = env.reset()
    depth = env.camera.data.output["distance_to_image_plane"]
    local_pose = torch.cat(
        [env.camera.data.pos_w - env.scene.env_origins, env.camera.data.quat_w_world], dim=1
    )
    dense, dense_valid = backproject_depth_world(depth, camera_metric_intrinsics(env.camera), local_pose, 2.0)
    valid_points = dense[0, dense_valid[0]]
    print(
        "[SMOKE] reset diagnostics",
        "q=", env.robot.data.joint_pos[0, env.joint_ids].detach().cpu().round(decimals=3).tolist(),
        "camera=", (env.camera.data.pos_w[0] - env.scene.env_origins[0]).detach().cpu().round(decimals=3).tolist(),
        "depth_min=", float(torch.where(depth > 0, depth, torch.full_like(depth, float("inf"))).amin()),
        "depth_lt2=", float(((depth > 0) & (depth < 2)).float().mean()),
        "frontend_valid=", env.valid_depth_ratio.detach().cpu().round(decimals=4).tolist(),
        "point_bounds=", (
            valid_points.amin(dim=0).detach().cpu().round(decimals=3).tolist(),
            valid_points.amax(dim=0).detach().cpu().round(decimals=3).tolist(),
        ),
    )
    assert observation["policy"].shape == (2, 173)
    assert observation["critic"].shape == (2, 179)
    for step in range(args_cli.steps):
        action = torch.zeros(2, 8, device=env.device)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(
            f"[SMOKE] step={step} reward={reward.mean().item():.4f} "
            f"coverage={env.coverage.mean().item():.4f} valid={env.valid_depth_ratio.mean().item():.4f} "
            f"chamfer={env.reconstruction_error.mean().item():.4f} collision={env.collision.tolist()} "
            f"terminated={terminated.tolist()}"
        )
        assert torch.isfinite(observation["policy"]).all()
        assert torch.isfinite(reward).all()
        assert terminated.shape == truncated.shape == (2,)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
