"""Orchestrate schema-v8 mixed-shape collection, one geometry per process."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from codex_zbot.tasks.layout import rotation_matrix_xyz,aabb_surface_clearance
from codex_zbot.tasks.shape_geometry import SHAPE_NAMES,shape_half_extent


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_).tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def completed_episodes(output: Path) -> int:
    return len(list(output.glob("ep_*.npz"))) if output.exists() else 0


def validate_latest(path: Path) -> dict[str, object]:
    manifest=json.loads((path.parent/'collection_manifest.json').read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as episode:
        required = {
            "object_centers", "object_sizes", "object_euler_xyz", "object_quaternions_wxyz",
            "object_bottom_heights", "object_visibility_per_object", "objects_in_view", "collision",
            "object_shape_ids","gt_object_points","minimum_surface_clearance_m",
        }
        missing = required.difference(episode.files)
        if missing:
            raise RuntimeError(f"{path.name} lacks schema-v8 fields: {sorted(missing)}")
        sizes = episode["object_sizes"]
        ids=episode['object_shape_ids']
        if ids.shape!=(len(sizes),) or not np.isin(ids,[0,1,2]).all():
            raise RuntimeError('invalid shape IDs')
        for size,shape in zip(sizes,ids,strict=True):
            if (shape==1 and not np.allclose(size,size[0])) or (shape==2 and not np.isclose(size[0],size[1])):
                raise RuntimeError('round primitive dimensions are inconsistent')
        if not manifest['objects'][0] <= len(sizes) <= manifest['objects'][1]:
            raise RuntimeError('out-of-range object count')
        quaternion = episode["object_quaternions_wxyz"]
        ranges=manifest['object_size_ranges_m']
        lower=np.asarray([ranges[a][0] for a in 'xyz'])
        upper=np.asarray([ranges[a][1] for a in 'xyz'])
        if not ((sizes >= lower - 1e-6).all() and (sizes <= upper + 1e-6).all()):
            raise RuntimeError(f"{path.name} contains an out-of-range object size")
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-4):
            raise RuntimeError(f"{path.name} contains a non-unit object quaternion")
        empty = episode["objects_in_view"] == 0
        empty_fraction = float(empty.mean())
        longest_empty = longest_true_run(empty)
        cumulative_visibility = episode["object_visibility_per_object"].max(axis=0)
        if empty_fraction > manifest['max_empty_frame_fraction'] or longest_empty > manifest['max_consecutive_empty_frames']:
            raise RuntimeError(
                f"{path.name} exceeds empty-frame limits: fraction={empty_fraction:.3f}, "
                f"longest_run={longest_empty}"
            )
        if float(cumulative_visibility.min()) < manifest['min_episode_per_object_visibility']:
            raise RuntimeError(
                f"{path.name} lacks cumulative evidence for an object: "
                f"{cumulative_visibility.tolist()}"
            )
        centers=episode['object_centers']
        half=np.stack([shape_half_extent(size,rotation_matrix_xyz(euler),shape)
                       for size,euler,shape in zip(sizes,episode['object_euler_xyz'],ids,strict=True)])
        gaps=[aabb_surface_clearance(centers[i],half[i],centers[j],half[j])
              for i in range(len(sizes)) for j in range(i+1,len(sizes))]
        if gaps and min(gaps)<float(episode['minimum_surface_clearance_m'])-1e-6:
            raise RuntimeError('scene violates sampled surface clearance')
        if 'maximum_nearest_surface_clearance_m' in episode and len(sizes)>1:
            radius=sizes.min(axis=1)*.5
            upper=np.maximum(np.linalg.norm(centers[:,None]-centers[None,:],axis=2)
                             -radius[:,None]-radius[None,:],0)
            np.fill_diagonal(upper,np.inf)
            if (upper.min(axis=1)>float(episode['maximum_nearest_surface_clearance_m'])+1e-6).any():
                raise RuntimeError('nearest-neighbor gap exceeds configured range')
        for axis,key in enumerate(['scene_object_x_range_m','scene_object_y_range_m']):
            bounds=manifest[key]
            if ((centers[:,axis]-half[:,axis]<bounds[0]-1e-6) |
                (centers[:,axis]+half[:,axis]>bounds[1]+1e-6)).any():
                raise RuntimeError('object exceeds work volume')
        if bool(episode['collision'].any()):
            raise RuntimeError('collision frames cannot be accepted')
        return {
            "episode": path.name,
            "objects": int(len(sizes)),
            "shapes":[SHAPE_NAMES[int(i)] for i in ids],
            "minimum_aabb_surface_gap_m":min(gaps) if gaps else None,
            "size_min_m": sizes.min(axis=0).tolist(),
            "size_max_m": sizes.max(axis=0).tolist(),
            "floating": int((episode["object_bottom_heights"] > 1e-5).sum()),
            "minimum_visibility": float(episode["object_visibility_per_object"].min()),
            "minimum_cumulative_visibility": float(cumulative_visibility.min()),
            "empty_frame_fraction": empty_fraction,
            "longest_empty_run": longest_empty,
            "collision_frames": int(episode["collision"].sum()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=__import__('sys').executable)
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "sequences_v8_random_shapes"))
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--objects_min",type=int,default=3)
    parser.add_argument("--objects_max",type=int,default=5)
    parser.add_argument("--shape_types",nargs="+",choices=SHAPE_NAMES,default=list(SHAPE_NAMES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop_after", type=int, default=0, help="Collect at most this many new episodes; 0 is unlimited.")
    args = parser.parse_args()

    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".collection.lock"
    owns_lock=False
    try:
        with lock.open("x", encoding="utf-8") as handle:
            owns_lock=True
            handle.write(json.dumps({"argv": sys.argv}))
        start = completed_episodes(output)
        if start >= args.episodes:
            print(f"[RANDOM-GEOMETRY] already complete: {start}/{args.episodes}", flush=True)
            return
        process_limit = args.episodes if args.stop_after <= 0 else min(args.episodes, start + args.stop_after)
        while completed_episodes(output) < process_limit:
            completed = completed_episodes(output)
            command = [
                args.python,
                str(ROOT / "scripts" / "collect_sequences.py"),
                "--headless", "--enable_cameras", "--device", args.device,
                "--out_dir", str(output), "--episodes", str(args.episodes),
                "--frames", str(args.frames), "--objects_min", str(args.objects_min), "--objects_max", str(args.objects_max),
                "--seed", str(args.seed), "--pose_attempts", "140",
                "--shape_types",*args.shape_types,
            ]
            if completed > 0:
                command.append("--resume")
            print(f"[RANDOM-GEOMETRY] launch episode {completed}/{args.episodes}", flush=True)
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Isaac collection process failed with exit code {result.returncode}")
            current = completed_episodes(output)
            if current != completed + 1:
                raise RuntimeError(f"expected one new episode, found {current - completed}")
            audit = validate_latest(output / f"ep_{current - 1:04d}.npz")
            print(f"[RANDOM-GEOMETRY] accepted {json.dumps(audit)} progress={current}/{args.episodes}", flush=True)
    finally:
        if owns_lock and lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
