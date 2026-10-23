"""Collect schema-v3 V4 PPO trajectories for alternating fine-tuning."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsl_rl.runners import OnPolicyRunner
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", default=str(ROOT / "configs/formal_v4.json"))
parser.add_argument("--drift_config", default=str(ROOT / "piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json"))
parser.add_argument("--drift_checkpoint", required=True)
parser.add_argument("--world_checkpoint", required=True)
parser.add_argument("--ppo_checkpoint", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=4)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from codex_zbot.config import load_config
from codex_zbot.policy.rewards import depth_consistent_visibility, points_inside_image
from codex_zbot.policy.rsl_rl_cfg import ActiveVisionPPORunnerCfg
from codex_zbot.tasks.active_vision_v4_env import ActiveVisionV4Env, ActiveVisionV4EnvCfg
from codex_zbot.tasks.scene import OBJECT_SIZES,camera_metric_intrinsics


FRAME_KEYS = ["rgb", "depth", "semantic", "pose", "joint_pos", "joint_vel", "collision", "action"]


def empty_record() -> dict[str, list]:
    record = {key: [] for key in FRAME_KEYS}
    record.update({"object_visibility": [], "object_visibility_per_object": [], "object_fov_containment_per_object": []})
    return record


def scene_metadata(env: ActiveVisionV4Env, env_id: int) -> dict[str, np.ndarray]:
    active = env.object_active[env_id].nonzero(as_tuple=False).squeeze(-1)
    return {
        "object_centers": env.object_position[env_id, active].detach().cpu().numpy().copy(),
        "object_sizes": np.asarray([OBJECT_SIZES[index] for index in active.tolist()], dtype=np.float32),
        "object_yaws": env.object_yaw[env_id, active].detach().cpu().numpy().copy(),
        "gt_points": env.gt_volume_cloud[env_id].detach().cpu().numpy().copy(),
        "active_ids": active.detach().cpu().numpy().copy(),
    }


def main() -> None:
    output = Path(args_cli.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args_cli.config)
    env_cfg = ActiveVisionV4EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = cfg.ppo.seed + 20_000
    env_cfg.sim.device = args_cli.device
    env_cfg.algorithm_config_path = str(Path(args_cli.config).resolve())
    env_cfg.drift_config_path = str(Path(args_cli.drift_config).resolve())
    env_cfg.drift_checkpoint = str(Path(args_cli.drift_checkpoint).resolve())
    env_cfg.world_checkpoint = str(Path(args_cli.world_checkpoint).resolve())
    env_cfg.decimation = cfg.ppo.decimation
    env_cfg.episode_length_s = cfg.ppo.episode_length_s
    env_cfg.sim.dt = cfg.ppo.physics_dt
    env_cfg.sim.render_interval = env_cfg.decimation
    raw = ActiveVisionPPORunnerCfg()
    raw.actor.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    raw.critic.rnn_hidden_dim = cfg.ppo.recurrent_hidden_dim
    agent_cfg = handle_deprecated_rsl_rl_cfg(raw, importlib.metadata.version("rsl-rl-lib"))
    env = ActiveVisionV4Env(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(str(Path(args_cli.ppo_checkpoint).resolve()))
    policy = runner.get_inference_policy(device=agent_cfg.device)
    observation = wrapped.get_observations()
    np.save(output / "intrinsics.npy", camera_metric_intrinsics(env.camera)[0].cpu().numpy())
    records = [empty_record() for _ in range(args_cli.num_envs)]
    scenes = [scene_metadata(env, index) for index in range(args_cli.num_envs)]
    completed = 0
    with torch.inference_mode():
        while completed < args_cli.episodes:
            rgb = env.camera.data.output["rgb"][..., :3]
            depth = env.camera.data.output["distance_to_image_plane"]
            semantic = env.camera.data.output["semantic_segmentation"]
            pose = torch.cat(
                [env.camera.data.pos_w - env.scene.env_origins, env.camera.data.quat_w_world], dim=1
            )
            joint_pos = env.robot.data.joint_pos[:, env.joint_ids]
            joint_vel = env.robot.data.joint_vel[:, env.joint_ids]
            action = policy(observation)
            intrinsics = camera_metric_intrinsics(env.camera)
            per_surface = env.cfg.gt_surface_points // env.cfg.max_objects
            # Visibility blocks are rebuilt per active object with an equal count.
            visibility_rows, containment_rows = [], []
            for env_id in range(args_cli.num_envs):
                count = len(scenes[env_id]["active_ids"])
                block = env.cfg.gt_surface_points // count
                used = block * count
                visible = depth_consistent_visibility(
                    env.gt_surface_cloud[env_id : env_id + 1, :used],
                    depth[env_id : env_id + 1],
                    intrinsics[env_id : env_id + 1],
                    pose[env_id : env_id + 1],
                )[0].reshape(count, block).float().mean(dim=1)
                inside = points_inside_image(
                    env.gt_surface_cloud[env_id : env_id + 1, :used],
                    intrinsics[env_id : env_id + 1],
                    pose[env_id : env_id + 1],
                    depth.shape[1],
                    depth.shape[2],
                    margin_px=0.05 * depth.shape[1],
                )[0].reshape(count, block).float().mean(dim=1)
                visibility_rows.append(visible.cpu().numpy())
                containment_rows.append(inside.cpu().numpy())
                record = records[env_id]
                record["rgb"].append(rgb[env_id].cpu().numpy().copy())
                record["depth"].append(depth[env_id].cpu().numpy().copy())
                record["semantic"].append(semantic[env_id].cpu().numpy().copy())
                record["pose"].append(pose[env_id].cpu().numpy().copy())
                record["joint_pos"].append(joint_pos[env_id].cpu().numpy().copy())
                record["joint_vel"].append(joint_vel[env_id].cpu().numpy().copy())
                record["collision"].append(bool(env.collision[env_id]))
                record["action"].append(action[env_id].cpu().numpy().copy())
                record["object_visibility"].append(float(visible.mean()))
                record["object_visibility_per_object"].append(visibility_rows[-1])
                record["object_fov_containment_per_object"].append(containment_rows[-1])

            pre_step_coverage = env.coverage.clone()
            observation, _, done, _ = wrapped.step(action)
            policy.reset(done)
            for env_id in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                # A short episode that did not reach the coverage success
                # condition is an unsafe terminal transition (contact or
                # joint-limit violation).  Associate it with the action that
                # caused termination so the World Model receives positives.
                if (
                    len(records[env_id]["collision"]) < env.max_episode_length
                    and float(pre_step_coverage[env_id]) < cfg.reward.success_coverage
                ):
                    records[env_id]["collision"][-1] = True
                if completed < args_cli.episodes and len(records[env_id]["rgb"]) >= 23:
                    arrays = {key: np.asarray(value) for key, value in records[env_id].items()}
                    scene = scenes[env_id]
                    arrays.update({key: value for key, value in scene.items() if key != "active_ids"})
                    arrays["scene_attempt_id"] = np.asarray(completed, dtype=np.int64)
                    arrays["camera_min_object_distance"] = np.asarray(
                        np.linalg.norm(arrays["pose"][:, None, :3] - scene["object_centers"][None], axis=-1).min(axis=1),
                        dtype=np.float32,
                    )
                    np.savez_compressed(output / f"ep_{completed:04d}.npz", **arrays)
                    completed += 1
                    print(f"[V4-REPLAY] episode={completed}/{args_cli.episodes} frames={len(arrays['rgb'])}", flush=True)
                records[env_id] = empty_record()
                scenes[env_id] = scene_metadata(env, env_id)
    manifest = {
        "schema_version": 3,
        "episodes": completed,
        "policy_checkpoint": str(Path(args_cli.ppo_checkpoint).resolve()),
        "drift_checkpoint": str(Path(args_cli.drift_checkpoint).resolve()),
        "world_checkpoint": str(Path(args_cli.world_checkpoint).resolve()),
        "seed": env_cfg.seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
