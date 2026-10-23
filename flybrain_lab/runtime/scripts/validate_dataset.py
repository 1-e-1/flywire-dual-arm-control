"""Validate sequence schema, first-frame visibility and numerical integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--min_first_valid_ratio", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("ep_*.npz"))
    required = {"rgb", "depth", "semantic", "pose", "joint_pos", "gt_points"}
    failures = []
    statistics = []
    if not (data_dir / "intrinsics.npy").exists():
        failures.append("missing intrinsics.npy")
    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            missing = required - set(episode.files)
            if missing:
                failures.append(f"{path.name}: missing {sorted(missing)}")
                continue
            frame_count = episode["rgb"].shape[0]
            if any(episode[key].shape[0] != frame_count for key in ["depth", "semantic", "pose", "joint_pos"]):
                failures.append(f"{path.name}: modalities have inconsistent frame counts")
            depth = episode["depth"]
            valid = np.isfinite(depth) & (depth > 0.0) & (depth <= args.max_depth)
            first_valid = float(valid[0].mean())
            informative = float((valid.reshape(frame_count, -1).mean(axis=1) >= args.min_first_valid_ratio).mean())
            if first_valid < args.min_first_valid_ratio:
                failures.append(f"{path.name}: first_valid={first_valid:.4f}")
            if not np.isfinite(episode["pose"]).all() or not np.isfinite(episode["gt_points"]).all():
                failures.append(f"{path.name}: NaN/Inf in pose or GT")
            statistics.append({"episode": path.name, "frames": frame_count, "first_valid": first_valid, "informative": informative})
    summary = {
        "data_dir": str(data_dir.resolve()),
        "episodes": len(files),
        "mean_first_valid": float(np.mean([item["first_valid"] for item in statistics])) if statistics else 0.0,
        "mean_informative_frame_fraction": float(np.mean([item["informative"] for item in statistics])) if statistics else 0.0,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "episodes": statistics}, indent=2), encoding="utf-8")
    if not files or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

