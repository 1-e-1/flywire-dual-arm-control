"""Run a checkpoint manifest through deterministic PPO evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = ROOT / "logs" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in manifest["experiments"]:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "evaluate_ppo.py"),
            "--headless",
            "--enable_cameras",
            "--config",
            experiment["config"],
            "--cfm_checkpoint",
            experiment["cfm_checkpoint"],
            "--belief_checkpoint",
            experiment["belief_checkpoint"],
            "--ppo_checkpoint",
            experiment["ppo_checkpoint"],
            "--num_envs",
            str(args.num_envs),
            "--episodes",
            str(args.episodes),
            "--output",
            str(output_dir / f"{experiment['name']}.json"),
        ]
        print("[ABLATION]", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

