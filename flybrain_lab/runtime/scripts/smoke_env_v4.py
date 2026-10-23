"""Two-step integration smoke test for the V4 IsaacLab environment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", default=str(ROOT / "configs/formal_v4.json"))
parser.add_argument("--drift_config", default=str(ROOT / "piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json"))
parser.add_argument("--drift_checkpoint", required=True)
parser.add_argument("--world_checkpoint", required=True)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument(
    "--print_semantic_map",
    action="store_true",
    help="Print Isaac semantic ID-to-label metadata for segmentation integration checks.",
)
args_cli = parser.parse_args()
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import torch

from codex_zbot.policy.features_v4 import V4_ACTOR_OBSERVATION_DIM, V4_CRITIC_OBSERVATION_DIM
from codex_zbot.tasks.active_vision_v4_env import ActiveVisionV4Env, ActiveVisionV4EnvCfg


def main() -> None:
    cfg = ActiveVisionV4EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    cfg.drift_config_path = str(Path(args_cli.drift_config).resolve())
    cfg.drift_checkpoint = str(Path(args_cli.drift_checkpoint).resolve())
    cfg.world_checkpoint = str(Path(args_cli.world_checkpoint).resolve())
    env = ActiveVisionV4Env(cfg)
    observation, _ = env.reset()
    assert observation["policy"].shape == (1, V4_ACTOR_OBSERVATION_DIM)
    assert observation["critic"].shape == (1, V4_CRITIC_OBSERVATION_DIM)
    if args_cli.print_semantic_map:
        print(f"[V4-SEMANTICS] {env.camera.data.info[0].get('semantic_segmentation')}", flush=True)
    for step in range(args_cli.steps):
        observation, reward, terminated, truncated, _ = env.step(torch.zeros(1, 8, device=env.device))
        assert torch.isfinite(observation["policy"]).all()
        assert torch.isfinite(observation["critic"]).all()
        assert torch.isfinite(reward).all()
        print(
            f"[V4-SMOKE] step={step} reward={float(reward.mean()):.4f} "
            f"coverage={float(env.coverage.mean()):.4f} chamfer={float(env.reconstruction_error.mean()):.4f} "
            f"unknown={float(env.unknown_fraction.mean()):.4f} clearance={float(env.camera_clearance[0]):.4f} "
            f"camera_xyz={(env.camera.data.pos_w[0] - env.scene.env_origins[0]).tolist()} "
            f"shield={bool(env.safety_shield_active[0])} "
            f"view_shield={bool(env.view_shield_active[0])} "
            f"surface_visibility={float(env.current_surface_visibility[0]):.4f} "
            f"object_pixel_ratio={float(env.object_pixel_ratio[0]):.4f} "
            f"per_object_pixels={env.object_pixel_ratio_per_object[0, env.object_active[0]].tolist()} "
            f"minimum_object_pixels={float(env.minimum_object_pixel_ratio[0]):.4f} "
            f"joint_speed={float(env.robot.data.joint_vel[0, env.joint_ids].norm()):.6f} "
            f"camera_unsafe={bool(env.camera_distance_unsafe[0] or env.camera_workcell_unsafe[0] or env.camera_height_unsafe[0])} "
            f"collision={bool(env.collision[0])}",
            flush=True,
        )
        assert terminated.shape == truncated.shape == (1,)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
