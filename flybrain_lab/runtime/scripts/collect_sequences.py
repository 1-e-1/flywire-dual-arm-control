"""Collect sequential RGB-D-S trajectories with a guaranteed informative first frame."""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--out_dir", default=str(ROOT / "data" / "sequences_v8_random_shapes"))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--objects_min", type=int, default=3)
parser.add_argument("--objects_max", type=int, default=5)
parser.add_argument("--shape_types", nargs="+", choices=["cuboid","sphere","cylinder"],
                    default=["cuboid","sphere","cylinder"])
parser.add_argument("--max_scene_attempts_per_episode",type=int,default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_valid_ratio", type=float, default=0.15)
parser.add_argument("--min_object_visible_ratio", type=float, default=0.02)
parser.add_argument("--pose_attempts", type=int, default=80)
parser.add_argument("--joint_range", type=float, default=1.4)
parser.add_argument("--depth_noise_std", type=float, default=0.002)
parser.add_argument("--max_depth_m",type=float,default=4.0,
                    help="preserve measured background rays for free-space evidence")
parser.add_argument("--decimation", type=int, default=4)
parser.add_argument("--camera_width", type=int, default=256)
parser.add_argument("--camera_height", type=int, default=256)
parser.add_argument("--horizontal_fov_deg", type=float, default=110.0)
parser.add_argument("--scene_x_min", type=float, default=0.54)
parser.add_argument("--scene_x_max", type=float, default=0.94)
parser.add_argument("--scene_y_min", type=float, default=-0.50)
parser.add_argument("--scene_y_max", type=float, default=0.50)
parser.add_argument("--min_object_surface_clearance", "--min_object_xy_clearance", dest="min_object_surface_clearance", type=float, default=0.06)
parser.add_argument("--max_object_surface_clearance", "--max_object_xy_clearance", dest="max_object_surface_clearance", type=float, default=0.12)
parser.add_argument("--max_nearest_surface_clearance",type=float,default=.45)
parser.add_argument("--object_grounded_probability", type=float, default=0.25)
parser.add_argument("--floating_bottom_min", type=float, default=0.06)
parser.add_argument("--floating_bottom_max", type=float, default=0.38)
parser.add_argument("--size_x_min", type=float, default=0.06)
parser.add_argument("--size_x_max", type=float, default=0.16)
parser.add_argument("--size_y_min", type=float, default=0.06)
parser.add_argument("--size_y_max", type=float, default=0.16)
parser.add_argument("--size_z_min", type=float, default=0.08)
parser.add_argument("--size_z_max", type=float, default=0.24)
parser.add_argument("--max_roll_pitch_deg", type=float, default=35.0)
parser.add_argument("--min_per_object_visible_ratio", type=float, default=0.03)
parser.add_argument("--min_initial_visible_objects", type=int, default=1)
parser.add_argument("--min_view_object_visible_ratio", type=float, default=0.002)
parser.add_argument("--min_episode_per_object_visibility", type=float, default=0.0)
parser.add_argument("--max_empty_frame_fraction", type=float, default=0.20)
parser.add_argument("--max_consecutive_empty_frames", type=int, default=15)
parser.add_argument("--min_camera_object_distance", type=float, default=0.65)
parser.add_argument("--min_object_fov_containment", type=float, default=0.98)
parser.add_argument("--image_margin_ratio", type=float, default=0.05)
parser.add_argument("--collision_force_threshold", type=float, default=20.0)
parser.add_argument("--view_change_interval", type=int, default=15)
parser.add_argument("--view_change_attempts", type=int, default=48)
parser.add_argument("--view_change_joint_delta", type=float, default=0.35)
parser.add_argument("--min_view_joint_change_l2", type=float, default=0.45)
parser.add_argument("--min_view_camera_translation", type=float, default=0.06)
parser.add_argument("--min_view_camera_rotation_deg", type=float, default=12.0)
parser.add_argument("--min_view_change_success_fraction", type=float, default=0.5)
parser.add_argument("--resume", action="store_true")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera, ContactSensor

from codex_zbot.tasks.scene import (
    camera_cfg,
    configure_camera_intrinsics,
    camera_metric_intrinsics,
    contact_cfg,
    object_cfg,
    robot_cfg,
)
from codex_zbot.tasks.layout import (
    SpacedLayoutConfig,
    quaternion_wxyz_from_euler_xyz,
    rotation_matrix_xyz,
    sample_spaced_zero_gravity_layout_6d,
)
from codex_zbot.tasks.shape_geometry import (
    SHAPE_NAMES,sample_shapes_and_sizes,sample_local_surface,shape_half_extent,
)
from codex_zbot.policy.rewards import depth_consistent_visibility, points_inside_image, visibility_by_object
from codex_zbot.utils.seed import seed_everything

GROUND_POINT_COUNT = 2048
OBJECT_SURFACE_POINT_COUNT = 1536


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run in a one-dimensional bool mask."""
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_).tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def sample_surface(center, size, euler_xyz, count, generator, device,shape_id=0):
    seed=int(torch.randint(0,2**31-1,(),generator=generator))
    points=torch.from_numpy(sample_local_surface(size.cpu().numpy(),shape_id,count,
                            np.random.default_rng(seed))).to(device)
    rotation = torch.from_numpy(rotation_matrix_xyz(np.asarray(euler_xyz, dtype=np.float32))).to(
        device=device, dtype=points.dtype
    )
    points = points @ rotation.T
    return points + center


def randomize_scene(
    objects,
    episode_seed,
    object_count,
    device,
    generator,
    scene_x_range,
    scene_y_range,
    min_object_xy_clearance,
    object_grounded_probability,
    floating_bottom_range,
    object_sizes,
    max_roll_pitch_rad,
    object_shape_ids,
    max_nearest_surface_clearance_m=.45,
):
    """Randomize a fixed object pool without changing the live PhysX topology."""
    rng = np.random.default_rng(episode_seed)
    specs = []
    # Identity, dimensions and shape stay fixed across rejected placements.
    # Otherwise visibility rejection would silently resample easier shapes.
    active_id_list = list(range(object_count))
    active_ids = set(active_id_list)
    active_sizes = np.asarray([object_sizes[index] for index in active_id_list], dtype=np.float32)
    active_centers, active_euler, active_grounded = sample_spaced_zero_gravity_layout_6d(
        active_sizes,
        rng,
        SpacedLayoutConfig(
            x_range_m=scene_x_range,
            y_range_m=scene_y_range,
            min_surface_clearance_m=min_object_xy_clearance,
            grounded_probability=object_grounded_probability,
            floating_bottom_range_m=floating_bottom_range,
            max_roll_pitch_rad=max_roll_pitch_rad,
            ensure_floating_object=False,
            max_nearest_surface_clearance_m=max_nearest_surface_clearance_m,
        ),
        shape_ids=np.asarray(object_shape_ids)[active_id_list],
    )
    layout_by_object = {
        object_id: (
            active_centers[slot],
            active_euler[slot],
            quaternion_wxyz_from_euler_xyz(active_euler[slot]),
            bool(active_grounded[slot]),
        )
        for slot, object_id in enumerate(active_id_list)
    }
    for index, obj in enumerate(objects):
        size = np.asarray(object_sizes[index], dtype=np.float32)
        active = index in active_ids
        center, euler_xyz, quaternion, grounded = (
            layout_by_object[index]
            if active
            else (
                np.asarray([0.45, 0.0, -2.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                False,
            )
        )
        pose = torch.zeros(1, 7, device=device)
        pose[0, :3] = torch.from_numpy(center).to(device)
        pose[0, 3:] = torch.from_numpy(quaternion).to(device)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        if active:
            specs.append((center, size, euler_xyz, quaternion, grounded,int(object_shape_ids[index])))

    ground = torch.rand(GROUND_POINT_COUNT, 3, generator=generator).to(device)
    ground[:, 0] = 0.15 + (scene_x_range[1] + 0.08 - 0.15) * ground[:, 0]
    ground[:, 1] = scene_y_range[0] + (scene_y_range[1] - scene_y_range[0]) * ground[:, 1]
    ground[:, 2] = 0.0
    parts = [ground]
    for center, size, euler_xyz, _, _,shape_id in specs:
        parts.append(
            sample_surface(
                torch.from_numpy(center).to(device),
                torch.from_numpy(size).to(device),
                euler_xyz,
                OBJECT_SURFACE_POINT_COUNT,
                generator,
                device,
                shape_id,
            )
        )
    return specs, torch.cat(parts, dim=0)


def step_sim(sim, robot, camera, contact, steps=1):
    dt = sim.get_physics_dt()
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        camera.update(dt)
        contact.update(dt)


def find_informative_initial_pose(
    sim,
    robot,
    camera,
    contact,
    generator,
    object_surface,
    object_centers,
    min_ratio,
    min_object_ratio,
    min_per_object_ratio,
    min_visible_objects,
    image_margin_ratio,
    min_camera_distance,
    collision_force_threshold,
    joint_range,
    attempts,
):
    """Find a pose that sees objects, rather than merely seeing the ground."""
    best_object_ratio = -1.0
    best_depth_ratio = -1.0
    best_min_distance = -1.0
    best_per_object = None
    best_fov = None
    best_state = None
    for _ in range(attempts):
        q = robot.data.default_joint_pos.clone()
        q += ((torch.rand(q.shape, generator=generator) * 2.0 - 1.0).to(robot.device) * joint_range)
        q = q.clamp(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        step_sim(sim, robot, camera, contact, 8)
        depth = camera.data.output["distance_to_image_plane"][0]
        ratio = float((torch.isfinite(depth) & (depth > 0.0) & (depth < 2.0)).float().mean())
        pose = torch.cat([camera.data.pos_w, camera.data.quat_w_world], dim=1)
        visible = depth_consistent_visibility(
            object_surface.unsqueeze(0), depth.unsqueeze(0), camera_metric_intrinsics(camera), pose
        )
        object_ratio = float(visible.float().mean())
        per_object_ratio = visibility_by_object(visible, OBJECT_SURFACE_POINT_COUNT)[0]
        margin_px = image_margin_ratio * min(camera.cfg.height, camera.cfg.width)
        inside = points_inside_image(
            object_surface.unsqueeze(0),
            camera_metric_intrinsics(camera),
            pose,
            camera.cfg.height,
            camera.cfg.width,
            margin_px=margin_px,
        )
        fov_containment = visibility_by_object(inside, OBJECT_SURFACE_POINT_COUNT)[0]
        visible_object_count = int((per_object_ratio >= min_per_object_ratio).sum())
        camera_distance = torch.linalg.vector_norm(object_centers - pose[0, :3], dim=1)
        minimum_camera_distance = float(camera_distance.min())
        distance_ok = minimum_camera_distance >= min_camera_distance
        collision_force = float(contact.data.net_forces_w[0].norm(dim=-1).amax())
        collision_ok = collision_force < collision_force_threshold
        # Rank by the evidence actually available to an object-count-agnostic
        # reconstructor.  Do not reward knowing or retaining every GT object.
        score = (visible_object_count, object_ratio, ratio, float(fov_containment.mean()))
        best_score = (
            -1 if best_per_object is None else int((best_per_object >= min_per_object_ratio).sum()),
            best_object_ratio,
            best_depth_ratio,
            -1.0 if best_fov is None else float(best_fov.mean()),
        )
        if distance_ok and collision_ok and score > best_score:
            best_object_ratio = object_ratio
            best_depth_ratio = ratio
            best_min_distance = minimum_camera_distance
            best_per_object = per_object_ratio.clone()
            best_fov = fov_containment.clone()
            best_state = q.clone()
        if (
            distance_ok
            and collision_ok
            and ratio >= min_ratio
            and object_ratio >= min_object_ratio
            and visible_object_count >= min_visible_objects
        ):
            return ratio, object_ratio, per_object_ratio, fov_containment, minimum_camera_distance
    if best_state is not None:
        robot.write_joint_state_to_sim(best_state, torch.zeros_like(best_state))
        robot.set_joint_position_target(best_state)
        step_sim(sim, robot, camera, contact, 8)
    if best_state is None:
        empty = torch.empty(0, device=robot.device)
        return -1.0, -1.0, empty, empty, -1.0
    assert best_per_object is not None and best_fov is not None
    return best_depth_ratio, best_object_ratio, best_per_object, best_fov, best_min_distance


def try_informative_view_change(
    sim,
    robot,
    camera,
    contact,
    generator,
    object_surface,
    object_centers,
    min_object_ratio,
    min_camera_distance,
    collision_force_threshold,
    joint_delta,
    min_joint_change_l2,
    min_camera_translation,
    min_camera_rotation_deg,
    attempts,
):
    """Apply a local camera move only after rendering confirms scene retention."""
    previous = robot.data.joint_pos.clone()
    previous_pose = torch.cat([camera.data.pos_w, camera.data.quat_w_world], dim=1).clone()
    for _ in range(attempts):
        target = previous.clone()
        target += ((torch.rand(target.shape, generator=generator) * 2.0 - 1.0).to(robot.device) * joint_delta)
        target = target.clamp(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        robot.set_joint_position_target(target)
        step_sim(sim, robot, camera, contact, 8)
        depth = camera.data.output["distance_to_image_plane"][0]
        pose = torch.cat([camera.data.pos_w, camera.data.quat_w_world], dim=1)
        visible = depth_consistent_visibility(
            object_surface.unsqueeze(0), depth.unsqueeze(0), camera_metric_intrinsics(camera), pose
        )
        object_ratio = float(visible.float().mean())
        minimum_distance = torch.linalg.vector_norm(object_centers - pose[0, :3], dim=1).min()
        collision_force = contact.data.net_forces_w[0].norm(dim=-1).amax()
        joint_change_l2 = torch.linalg.vector_norm(target - previous, dim=1).min()
        camera_translation = torch.linalg.vector_norm(pose[0, :3] - previous_pose[0, :3])
        quaternion_dot = torch.sum(pose[0, 3:] * previous_pose[0, 3:]).abs().clamp(0.0, 1.0)
        camera_rotation_deg = torch.rad2deg(2.0 * torch.acos(quaternion_dot))
        view_change_ok = (
            float(joint_change_l2) >= min_joint_change_l2
            and (
                float(camera_translation) >= min_camera_translation
                or float(camera_rotation_deg) >= min_camera_rotation_deg
            )
        )
        if (
            object_ratio >= min_object_ratio
            and float(minimum_distance) >= min_camera_distance
            and float(collision_force) < collision_force_threshold
            and view_change_ok
        ):
            return True
    robot.write_joint_state_to_sim(previous, torch.zeros_like(previous))
    robot.set_joint_position_target(previous)
    step_sim(sim, robot, camera, contact, 8)
    return False


def main() -> None:
    if not math.isfinite(args_cli.max_depth_m) or args_cli.max_depth_m<=0:
        raise ValueError('max_depth_m must be finite and positive')
    operational_options={'out_dir','episodes','resume','headless','livestream','enable_cameras',
                         'device','cpu','kit_args','verbose','info','experience','rendering_mode','xr'}
    sampling_config={key:value for key,value in vars(args_cli).items() if key not in operational_options}
    sampling_config['pixel_coordinate_convention']='integer_indices_pixel_center_v1'
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "collection_manifest.json"
    existing = sorted(out_dir.glob("ep_*.npz"))
    if existing and not args_cli.resume:
        raise RuntimeError(f"{out_dir} already contains episodes; pass --resume or choose a new output directory")
    episode = len(existing)
    scene_attempt = 0
    if args_cli.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("schema_version") != 8 or previous_manifest.get("sampling_config") != sampling_config:
            raise RuntimeError("cannot mix old geometry/camera protocols or changed sampling settings; use a new directory")
        episode = int(previous_manifest["completed_episodes"])
        scene_attempt = int(previous_manifest["scene_attempts"])
        if episode != len(existing):
            raise RuntimeError("Collection manifest and episode files disagree; audit the dataset before resuming")
    elif existing:
        raise RuntimeError("cannot resume episodes without a compatible collection manifest")
    if not 1 <= args_cli.objects_min <= args_cli.objects_max or args_cli.max_scene_attempts_per_episode<1:
        raise ValueError("invalid object count or attempt budget")
    if not (
        args_cli.min_object_surface_clearance <= args_cli.max_object_surface_clearance
        and args_cli.size_x_min < args_cli.size_x_max
        and args_cli.size_y_min < args_cli.size_y_max
        and args_cli.size_z_min < args_cli.size_z_max
    ):
        raise ValueError("invalid random geometry ranges")
    geometry_rng = np.random.default_rng(args_cli.seed + episode * 1_000_003 + 71)
    pool_sizes,pool_shape_ids = sample_shapes_and_sizes(
        args_cli.objects_max,
        geometry_rng,
        (args_cli.size_x_min, args_cli.size_y_min, args_cli.size_z_min),
        (args_cli.size_x_max, args_cli.size_y_max, args_cli.size_z_max),
        args_cli.shape_types,
    )
    episode_min_xy_clearance = float(
        geometry_rng.uniform(args_cli.min_object_surface_clearance, args_cli.max_object_surface_clearance)
    )
    # Draw the episode complexity once.  Rejected camera/layout attempts must
    # not resample it, otherwise easier low-count scenes dominate the accepted
    # dataset even though the proposal distribution was uniform.
    object_count_rng = np.random.default_rng(args_cli.seed + episode * 1_000_003 + 193)
    episode_object_count = int(object_count_rng.integers(args_cli.objects_min, args_cli.objects_max + 1))
    # Keep the original proposal even if no valid trajectory is accepted.
    # Failed complex scenes must remain auditable, not silently disappear.
    proposal={'episode':episode,'object_count':episode_object_count,
              'geometry_seed':args_cli.seed+episode*1_000_003+71,
              'sizes':pool_sizes[:episode_object_count].tolist(),
              'shape_ids':pool_shape_ids[:episode_object_count].tolist(),
              'minimum_surface_clearance_m':episode_min_xy_clearance,
              'sampling_config':sampling_config}
    proposal_path=out_dir/f'geometry_{episode:04d}.json'
    if proposal_path.exists():
        if json.loads(proposal_path.read_text(encoding='utf-8')) != proposal:
            raise RuntimeError('existing geometry proposal differs; use a new output directory')
    else:
        proposal_path.write_text(json.dumps(proposal,indent=2),encoding='utf-8')
    # Physics parses non-uniform rigid-body scale only before simulation starts.
    # Each process therefore collects exactly one accepted continuous-geometry episode.
    process_episode_stop = min(args_cli.episodes, episode + 1)
    seed_everything(args_cli.seed)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    )
    sim.set_camera_view([1.5, -1.5, 1.2], [0.45, 0.0, 0.2])
    # Local procedural ground: no remote USD dependency; top is exactly z=0.
    floor=sim_utils.CuboidCfg(size=(4.,4.,.02),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.5,.5,.5)))
    floor.func("/World/ground",floor,translation=(0.,0.,-.01))
    light = sim_utils.DomeLightCfg(intensity=1800.0)
    light.func("/World/Light", light)
    robot = Articulation(robot_cfg("/World/Robot"))
    camera = Camera(camera_cfg("/World/Robot/b8/Camera", args_cli.camera_width, args_cli.camera_height))
    contact = ContactSensor(contact_cfg("/World/Robot/.*"))
    objects = [
        RigidObject(object_cfg(index, f"/World/object_{index}", tuple(pool_sizes[index].tolist()),
                               SHAPE_NAMES[pool_shape_ids[index]]))
        for index in range(args_cli.objects_max)
    ]
    sim.reset()
    configured_intrinsics = configure_camera_intrinsics(camera, args_cli.horizontal_fov_deg)
    step_sim(sim, robot, camera, contact, 5)
    np.save(out_dir / "intrinsics.npy", configured_intrinsics[0].cpu().numpy())

    process_start_attempt=scene_attempt
    while episode < process_episode_stop:
        if scene_attempt-process_start_attempt >= args_cli.max_scene_attempts_per_episode:
            raise RuntimeError("scene attempt budget exhausted; preserve this geometry and inspect rejection logs")
        scene_attempt += 1
        # Attempt-local RNG makes a resumed collection bitwise reproducible
        # without replaying every rejected scene from the beginning.
        generator = torch.Generator(device="cpu").manual_seed(
            args_cli.seed + scene_attempt * 104729
        )
        object_count = episode_object_count
        specs, gt_points = randomize_scene(
            objects,
            args_cli.seed + scene_attempt * 1009,
            object_count,
            sim.device,
            generator,
            (args_cli.scene_x_min, args_cli.scene_x_max),
            (args_cli.scene_y_min, args_cli.scene_y_max),
            episode_min_xy_clearance,
            args_cli.object_grounded_probability,
            (args_cli.floating_bottom_min, args_cli.floating_bottom_max),
            pool_sizes,
            math.radians(args_cli.max_roll_pitch_deg),
            pool_shape_ids,
            args_cli.max_nearest_surface_clearance,
        )
        step_sim(sim, robot, camera, contact, 4)
        object_centers = torch.stack([torch.from_numpy(item[0]).to(sim.device) for item in specs])
        valid_ratio, object_visible_ratio, per_object_ratio, fov_containment, min_camera_distance = (
            find_informative_initial_pose(
            sim,
            robot,
            camera,
            contact,
            generator,
            gt_points[GROUND_POINT_COUNT:],
            object_centers,
            args_cli.min_valid_ratio,
            args_cli.min_object_visible_ratio,
            args_cli.min_per_object_visible_ratio,
            args_cli.min_initial_visible_objects,
            args_cli.image_margin_ratio,
            args_cli.min_camera_object_distance,
            args_cli.collision_force_threshold,
            args_cli.joint_range,
            args_cli.pose_attempts,
            )
        )
        visible_object_count = int((per_object_ratio >= args_cli.min_per_object_visible_ratio).sum())
        if (
            valid_ratio < args_cli.min_valid_ratio
            or object_visible_ratio < args_cli.min_object_visible_ratio
            or min_camera_distance < args_cli.min_camera_object_distance
            or visible_object_count < args_cli.min_initial_visible_objects
        ):
            print(
                f"[COLLECT] reject scene attempt={scene_attempt}, best first-frame valid={valid_ratio:.3f} "
                f"object_visible={object_visible_ratio:.3f} per_object={per_object_ratio.tolist()} "
                f"fov_containment={fov_containment.tolist()} "
                f"min_distance={min_camera_distance:.3f}"
            )
            continue

        records = {
            key: []
            for key in [
                "rgb",
                "depth",
                "semantic",
                "pose",
                "joint_pos",
                "joint_vel",
                "collision",
                "object_visibility",
                "object_visibility_per_object",
                "object_fov_containment_per_object",
                "objects_in_view",
                "objects_fully_in_frame",
                "camera_min_object_distance",
            ]
        }
        view_change_frames = []
        view_change_accepted = []
        for frame in range(args_cli.frames):
            if frame > 0 and frame % args_cli.view_change_interval == 0:
                accepted = try_informative_view_change(
                    sim,
                    robot,
                    camera,
                    contact,
                    generator,
                    gt_points[GROUND_POINT_COUNT:],
                    object_centers,
                    args_cli.min_view_object_visible_ratio,
                    args_cli.min_camera_object_distance,
                    args_cli.collision_force_threshold,
                    args_cli.view_change_joint_delta,
                    args_cli.min_view_joint_change_l2,
                    args_cli.min_view_camera_translation,
                    args_cli.min_view_camera_rotation_deg,
                    args_cli.view_change_attempts,
                )
                view_change_frames.append(frame)
                view_change_accepted.append(accepted)
                print(f"[COLLECT] frame={frame:03d} informative_view_change={accepted}")
            if frame > 0:
                step_sim(sim, robot, camera, contact, args_cli.decimation)
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
            depth = camera.data.output["distance_to_image_plane"][0].clone()
            valid = torch.isfinite(depth) & (depth > 0.0) & (depth < args_cli.max_depth_m)
            depth=torch.where(valid,depth,torch.zeros_like(depth))
            if args_cli.depth_noise_std > 0:
                noise = torch.randn(depth.shape, generator=generator).to(depth.device) * args_cli.depth_noise_std
                depth = torch.where(valid, (depth + noise).clamp_min(0.001), torch.zeros_like(depth))
            semantic = camera.data.output["semantic_segmentation"][0].cpu().numpy().copy()
            pose_tensor = torch.cat([camera.data.pos_w, camera.data.quat_w_world], dim=1)
            pose = pose_tensor[0].cpu().numpy()
            visible = depth_consistent_visibility(
                gt_points[GROUND_POINT_COUNT:].unsqueeze(0),
                depth.unsqueeze(0),
                camera_metric_intrinsics(camera),
                pose_tensor,
            )
            per_object_visible = visibility_by_object(visible, OBJECT_SURFACE_POINT_COUNT)[0]
            margin_px = args_cli.image_margin_ratio * min(camera.cfg.height, camera.cfg.width)
            inside = points_inside_image(
                gt_points[GROUND_POINT_COUNT:].unsqueeze(0),
                camera_metric_intrinsics(camera),
                pose_tensor,
                camera.cfg.height,
                camera.cfg.width,
                margin_px=margin_px,
            )
            per_object_fov = visibility_by_object(inside, OBJECT_SURFACE_POINT_COUNT)[0]
            camera_distance = torch.linalg.vector_norm(object_centers - pose_tensor[0, :3], dim=1)
            force = contact.data.net_forces_w[0].norm(dim=-1).amax()
            records["rgb"].append(rgb)
            records["depth"].append(depth.cpu().numpy())
            records["semantic"].append(semantic)
            records["pose"].append(pose)
            records["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            records["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            records["collision"].append(bool(force > args_cli.collision_force_threshold))
            records["object_visibility"].append(float(visible.float().mean()))
            records["object_visibility_per_object"].append(per_object_visible.cpu().numpy())
            records["object_fov_containment_per_object"].append(per_object_fov.cpu().numpy())
            records["objects_in_view"].append(int((per_object_visible >= args_cli.min_per_object_visible_ratio).sum()))
            records["objects_fully_in_frame"].append(
                int((per_object_fov >= args_cli.min_object_fov_containment).sum())
            )
            records["camera_min_object_distance"].append(float(camera_distance.min()))

        centers = np.stack([item[0] for item in specs])
        sizes = np.stack([item[1] for item in specs])
        euler_xyz = np.stack([item[2] for item in specs]).astype(np.float32)
        quaternions_wxyz = np.stack([item[3] for item in specs]).astype(np.float32)
        grounded = np.asarray([item[4] for item in specs], dtype=np.bool_)
        shape_ids=np.asarray([item[5] for item in specs],dtype=np.int32)
        yaws = euler_xyz[:, 2]
        aabb_half = np.stack(
            [shape_half_extent(size,rotation_matrix_xyz(angle),shape_id)
             for size, angle,shape_id in zip(sizes,euler_xyz,shape_ids,strict=True)]
        )
        bottom_heights = centers[:, 2] - aabb_half[:, 2]
        output = {key: np.asarray(value) for key, value in records.items()}
        output.update(
            gt_points=gt_points.cpu().numpy(),
            gt_object_points=gt_points[GROUND_POINT_COUNT:].cpu().numpy(),
            object_shape_ids=shape_ids,
            object_shape_names=np.asarray([SHAPE_NAMES[i] for i in shape_ids]),
            schema_version=np.asarray(8,dtype=np.int32),
            minimum_surface_clearance_m=np.asarray(episode_min_xy_clearance,dtype=np.float32),
            maximum_nearest_surface_clearance_m=np.asarray(args_cli.max_nearest_surface_clearance,dtype=np.float32),
            geometry_seed=np.asarray(args_cli.seed+episode*1_000_003+71,dtype=np.int64),
            object_centers=centers,
            object_sizes=sizes,
            object_yaws=yaws,
            object_euler_xyz=euler_xyz,
            object_quaternions_wxyz=quaternions_wxyz,
            object_grounded=grounded,
            object_bottom_heights=bottom_heights,
            view_change_frames=np.asarray(view_change_frames, dtype=np.int32),
            view_change_accepted=np.asarray(view_change_accepted, dtype=np.bool_),
            scene_attempt_id=np.asarray(scene_attempt, dtype=np.int64),
        )
        adjacent_joint_change = np.linalg.norm(np.diff(output["joint_pos"], axis=0), axis=1)
        empty_frames = output["objects_in_view"] == 0
        empty_frame_fraction = float(empty_frames.mean())
        consecutive_empty_frames = longest_true_run(empty_frames)
        cumulative_per_object_visibility = output["object_visibility_per_object"].max(axis=0)
        sequence_failures = []
        if float(cumulative_per_object_visibility.min()) < args_cli.min_episode_per_object_visibility:
            sequence_failures.append("insufficient_cumulative_object_evidence")
        if empty_frame_fraction > args_cli.max_empty_frame_fraction:
            sequence_failures.append("too_many_empty_frames")
        if consecutive_empty_frames > args_cli.max_consecutive_empty_frames:
            sequence_failures.append("empty_run_too_long")
        if bool(output["collision"].any()):
            sequence_failures.append("collision")
        if args_cli.frames > args_cli.view_change_interval and float(adjacent_joint_change.max()) < args_cli.min_view_joint_change_l2:
            sequence_failures.append("insufficient_joint_view_change")
        minimum_successes = math.ceil(len(view_change_accepted) * args_cli.min_view_change_success_fraction)
        if sum(view_change_accepted) < minimum_successes:
            sequence_failures.append("insufficient_accepted_view_changes")
        if sequence_failures:
            print(
                f"[COLLECT] reject completed trajectory scene_attempt={scene_attempt} reasons={sequence_failures} "
                f"min_objects_in_view={int(output['objects_in_view'].min())}/{object_count} "
                f"empty_fraction={empty_frame_fraction:.3f} longest_empty_run={consecutive_empty_frames} "
                f"cumulative_per_object={cumulative_per_object_visibility.tolist()} "
                f"max_joint_change_l2={float(adjacent_joint_change.max()):.3f}"
                f" accepted_view_changes={sum(view_change_accepted)}/{len(view_change_accepted)}"
            )
            continue
        path = out_dir / f"ep_{episode:04d}.npz"
        np.savez_compressed(path, **output)
        print(
            f"[COLLECT] episode={episode:04d} objects={object_count} first_valid={valid_ratio:.3f} "
            f"object_visible={object_visible_ratio:.3f} rgb={output['rgb'].shape} gt={output['gt_points'].shape}"
        )
        episode += 1
        manifest = {
            "schema_version": 8,
            "sampling_config":sampling_config,
            "shape_id_to_name":list(SHAPE_NAMES),
            "shape_sampling":"iid uniform categorical; dimensions conditional on primitive family",
            "camera_usd_focal_length":24.0,
            "ground":"procedural static slab, top z=0",
            "seed": args_cli.seed,
            "requested_episodes": args_cli.episodes,
            "completed_episodes": episode,
            "frames_per_episode": args_cli.frames,
            "physics_dt_s": 1.0 / 60.0,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "control_model": "direct_joint_kinematics",
            "decimation": args_cli.decimation,
            "control_frequency_hz": 60.0 / args_cli.decimation,
            "objects": [args_cli.objects_min, args_cli.objects_max],
            "min_valid_ratio": args_cli.min_valid_ratio,
            "min_object_visible_ratio": args_cli.min_object_visible_ratio,
            "pose_attempts": args_cli.pose_attempts,
            "joint_range_rad": args_cli.joint_range,
            "depth_noise_std_m": args_cli.depth_noise_std,
            "recorded_depth_max_m":args_cli.max_depth_m,
            "camera_resolution_wh": [args_cli.camera_width, args_cli.camera_height],
            "camera_horizontal_fov_deg": args_cli.horizontal_fov_deg,
            "camera_vertical_fov_deg": math.degrees(
                2.0
                * math.atan(
                    args_cli.camera_height
                    / args_cli.camera_width
                    * math.tan(math.radians(args_cli.horizontal_fov_deg) * 0.5)
                )
            ),
            "scene_object_x_range_m": [args_cli.scene_x_min, args_cli.scene_x_max],
            "scene_object_y_range_m": [args_cli.scene_y_min, args_cli.scene_y_max],
            "object_surface_clearance_range_m": [
                args_cli.min_object_surface_clearance,
                args_cli.max_object_surface_clearance,
            ],
            "episode_minimum_object_surface_clearance_m": episode_min_xy_clearance,
            "object_size_ranges_m": {
                "x": [args_cli.size_x_min, args_cli.size_x_max],
                "y": [args_cli.size_y_min, args_cli.size_y_max],
                "z": [args_cli.size_z_min, args_cli.size_z_max],
            },
            "episode_object_pool_sizes_m": pool_sizes.tolist(),
            "episode_object_pool_shape_ids":pool_shape_ids.tolist(),
            "clearance_meaning":"uniformly drawn minimum conservative AABB surface gap, not a fixed pairwise distance",
            "maximum_nearest_surface_clearance_m":args_cli.max_nearest_surface_clearance,
            "object_roll_pitch_range_deg": [-args_cli.max_roll_pitch_deg, args_cli.max_roll_pitch_deg],
            "object_yaw_range_deg": [-180.0, 180.0],
            "object_grounded_probability": args_cli.object_grounded_probability,
            "floating_object_bottom_range_m": [args_cli.floating_bottom_min, args_cli.floating_bottom_max],
            "min_per_object_visible_ratio": args_cli.min_per_object_visible_ratio,
            "initial_visibility_policy": "aggregate_evidence_without_object_count_prior",
            "min_initial_visible_objects": args_cli.min_initial_visible_objects,
            "min_view_object_visible_ratio": args_cli.min_view_object_visible_ratio,
            "min_episode_per_object_visibility": args_cli.min_episode_per_object_visibility,
            "max_empty_frame_fraction": args_cli.max_empty_frame_fraction,
            "max_consecutive_empty_frames": args_cli.max_consecutive_empty_frames,
            "min_object_fov_containment": args_cli.min_object_fov_containment,
            "image_margin_ratio": args_cli.image_margin_ratio,
            "min_camera_object_distance_m": args_cli.min_camera_object_distance,
            "collision_force_threshold_n": args_cli.collision_force_threshold,
            "view_change_interval_frames": args_cli.view_change_interval,
            "view_change_attempts": args_cli.view_change_attempts,
            "view_change_joint_delta_rad": args_cli.view_change_joint_delta,
            "min_view_joint_change_l2_rad": args_cli.min_view_joint_change_l2,
            "min_view_camera_translation_m": args_cli.min_view_camera_translation,
            "min_view_camera_rotation_deg": args_cli.min_view_camera_rotation_deg,
            "min_view_change_success_fraction": args_cli.min_view_change_success_fraction,
            "scene_attempts": scene_attempt,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # This collector does not use Replicator.  Immediate Kit shutdown avoids
        # a known long PhysX/Fabric teardown stall after large sequential runs;
        # every NPZ and the manifest are written synchronously before this point.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
