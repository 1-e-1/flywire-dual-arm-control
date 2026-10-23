"""Collect PPO trajectories as sequential replay for alternating CFM/world-model updates."""
from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", required=True)
parser.add_argument("--cfm_checkpoint", required=True)
parser.add_argument("--belief_checkpoint", required=True)
parser.add_argument("--ppo_checkpoint", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--num_envs", type=int, default=8)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_env import ActiveVisionEnv, ActiveVisionEnvCfg
from codex_zbot.tasks.scene import camera_metric_intrinsics


def empty_record():
    return {key: [] for key in ["rgb", "depth", "semantic", "pose", "joint_pos", "joint_vel", "collision"]}


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args_cli.config)
    env_cfg = ActiveVisionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = cfg.ppo.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.cfm_checkpoint = str(Path(args_cli.cfm_checkpoint).resolve())
    env_cfg.belief_checkpoint = str(Path(args_cli.belief_checkpoint).resolve())
    raw_agent_cfg = ActiveVisionPPORunnerCfg()
    raw_agent_cfg.actor.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    raw_agent_cfg.critic.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(raw_agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    env = ActiveVisionEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(str(Path(args_cli.ppo_checkpoint).resolve()))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observation = wrapped.get_observations()
    np.save(out_dir / "intrinsics.npy", camera_metric_intrinsics(env.camera)[0].cpu().numpy())

    records = [empty_record() for _ in range(args_cli.num_envs)]
    episode_gt = [env.gt_cloud[index].cpu().numpy().copy() for index in range(args_cli.num_envs)]
    completed = 0
    with torch.inference_mode():
        while completed < args_cli.episodes:
            rgb = env.camera.data.output["rgb"][..., :3].cpu().numpy()
            depth = env.camera.data.output["distance_to_image_plane"].cpu().numpy()
            semantic = env.camera.data.output["semantic_segmentation"].cpu().numpy()
            pose = torch.cat(
                [env.camera.data.pos_w - env.scene.env_origins, env.camera.data.quat_w_world], dim=1
            ).cpu().numpy()
            joint_position = env.robot.data.joint_pos[:, env.joint_ids].cpu().numpy()
            joint_velocity = env.robot.data.joint_vel[:, env.joint_ids].cpu().numpy()
            collision = env.collision.cpu().numpy()
            for env_id in range(args_cli.num_envs):
                records[env_id]["rgb"].append(rgb[env_id].copy())
                records[env_id]["depth"].append(depth[env_id].copy())
                records[env_id]["semantic"].append(semantic[env_id].copy())
                records[env_id]["pose"].append(pose[env_id].copy())
                records[env_id]["joint_pos"].append(joint_position[env_id].copy())
                records[env_id]["joint_vel"].append(joint_velocity[env_id].copy())
                records[env_id]["collision"].append(bool(collision[env_id]))

            action = policy(observation)
            observation, _, done, _ = wrapped.step(action)
            policy.reset(done)
            for env_id in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if completed < args_cli.episodes and len(records[env_id]["rgb"]) >= cfg.cfm.sequence_length:
                    payload = {key: np.asarray(value) for key, value in records[env_id].items()}
                    payload["gt_points"] = episode_gt[env_id]
                    np.savez_compressed(out_dir / f"ep_{completed:04d}.npz", **payload)
                    completed += 1
                    print(f"[REPLAY] saved episode={completed}/{args_cli.episodes} frames={len(payload['rgb'])}")
                records[env_id] = empty_record()
                episode_gt[env_id] = env.gt_cloud[env_id].cpu().numpy().copy()
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
