"""Static simulation and sim-to-real safety audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pxr import Usd, UsdPhysics

from codex_zbot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage = Usd.Stage.Open(str(PROJECT_ROOT / "zbot_8s_new.usd"))
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    errors = []
    warnings = []
    limits = []
    velocities = []
    for joint in joints:
        lower = float(joint.GetAttribute("physics:lowerLimit").Get())
        upper = float(joint.GetAttribute("physics:upperLimit").Get())
        velocity = float(joint.GetAttribute("physxJoint:maxJointVelocity").Get())
        limits.append((lower, upper))
        velocities.append(velocity)
    if len(joints) != 8:
        errors.append(f"Expected 8 revolute joints, found {len(joints)}")
    if any(lower >= upper for lower, upper in limits):
        errors.append("Invalid USD joint limits")
    control_dt = cfg.ppo.physics_dt * cfg.ppo.decimation
    maximum_commanded_speed = cfg.ppo.action_scale_rad / control_dt
    asset_speed = min(velocities) * math.pi / 180.0 if velocities else 0.0
    if maximum_commanded_speed >= asset_speed:
        errors.append("Commanded position slew can exceed USD velocity limit")
    if cfg.ppo.joint_limit_margin_rad < math.radians(5.0):
        errors.append("Joint safety margin is below 5 degrees")
    if cfg.ppo.action_filter_alpha <= 0.0 or cfg.ppo.action_filter_alpha > 1.0:
        errors.append("Action low-pass alpha must be in (0,1]")
    warnings.extend(
        [
            "Hardware emergency-stop wiring cannot be verified in simulation.",
            "Real communication watchdog and latency bounds require a hardware-in-the-loop test.",
            "Depth/semantic noise distributions must be calibrated against the physical sensor.",
            "Contact-force threshold must be converted to robot-specific torque/force limits before deployment.",
            "Domain randomization coverage does not prove sim-to-real safety.",
        ]
    )
    report = {
        "joints": len(joints),
        "joint_limits_deg": limits,
        "asset_velocity_limit_rad_s": asset_speed,
        "commanded_slew_rad_s": maximum_commanded_speed,
        "control_frequency_hz": 1.0 / control_dt,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

