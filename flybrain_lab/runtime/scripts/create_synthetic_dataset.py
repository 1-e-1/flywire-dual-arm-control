"""Create a tiny deterministic dataset for pipeline tests, not for research training."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/smoke_sequences")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    height = width = 32
    intrinsics = np.array([[32.0, 0.0, 15.5], [0.0, 32.0, 15.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.save(output / "intrinsics.npy", intrinsics)
    for episode in range(args.episodes):
        rgb = rng.integers(0, 255, size=(args.frames, height, width, 3), dtype=np.uint8)
        depth = np.full((args.frames, height, width, 1), 0.5, dtype=np.float32)
        depth += rng.normal(0.0, 0.002, depth.shape).astype(np.float32)
        semantic = np.ones((args.frames, height, width, 1), dtype=np.int32)
        pose = np.zeros((args.frames, 7), dtype=np.float32)
        pose[:, 3] = 1.0
        pose[:, 1] = np.linspace(-0.02, 0.02, args.frames)
        joint = rng.normal(0.0, 0.05, size=(args.frames, 8)).astype(np.float32)
        joint_velocity = np.zeros_like(joint)
        joint_velocity[1:] = (joint[1:] - joint[:-1]) * 15.0
        gt = rng.uniform([0.35, -0.25, 0.0], [0.70, 0.25, 0.35], size=(512, 3)).astype(np.float32)
        np.savez_compressed(
            output / f"ep_{episode:04d}.npz",
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            pose=pose,
            joint_pos=joint,
            joint_vel=joint_velocity,
            collision=np.zeros(args.frames, dtype=np.bool_),
            gt_points=gt,
        )
    print(output.resolve())


if __name__ == "__main__":
    main()

