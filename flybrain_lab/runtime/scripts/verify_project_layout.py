"""Read-only CPU migration audit: paths, hashes and strict checkpoint loading.

Run at the unified root. No simulator, collection, training or file output.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_layout():
    assert not (ROOT / "Codex").exists(), "Old nested project still exists"
    assert (ROOT / "zbot_8s_new.usd").is_file(), "Robot asset missing"
    manifest = json.loads((ROOT / "configs/retained_models.json").read_text(encoding="utf-8"))
    assert {"cfm", "drifting", "non_neural"} <= manifest.keys()
    for route in ("cfm", "drifting", "non_neural"):
        for key, value in manifest[route].items():
            if value and (key.endswith("checkpoint") or key in {"config", "controller_config", "training_data", "evaluation", "camera_config", "native_sample"}):
                assert (ROOT / value).exists(), f"Missing {route}.{key}: {value}"
    hashes = json.loads((ROOT / "maintenance/retained_weights.json").read_text(encoding="utf-8-sig"))
    for record in hashes:
        path = Path(record["destination"])
        # Ledger stores original absolute paths; use the retained relative suffix
        # so verification still works if the whole unified project is relocated.
        parts = str(record["destination"]).replace("\\", "/").split("/")
        path = ROOT.joinpath(*parts[parts.index("checkpoints"):])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest.upper() == record["sha256"], f"Changed checkpoint: {path}"
    for name in ("codex_zbot", "piaoyi_model", "classical_fusion", "scripts", "tests"):
        for file in (ROOT / name).rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
    return manifest, len(hashes)


def check_models(manifest, forward=False):
    import torch
    from codex_zbot.config import load_config
    from codex_zbot.cfm import CFMCondition, SecondOrderCFM
    from codex_zbot.world_model.model import BeliefLearningSystem
    from codex_zbot.utils.checkpoint import load_checkpoint
    from piaoyi_model.config import load_drift_config
    from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
    from piaoyi_model.belief_v4 import V4BeliefLearningSystem

    torch.set_num_threads(4)
    torch.manual_seed(20260906)
    cfm_cfg = load_config(ROOT / manifest["cfm"]["config"])
    drift_cfg = load_drift_config(ROOT / manifest["drifting"]["config"])
    controller_cfg = load_config(ROOT / manifest["drifting"]["controller_config"])
    models = [
        ("CFM", SecondOrderCFM(cfm_cfg.cfm), manifest["cfm"]["checkpoint"]),
        ("CFM world", BeliefLearningSystem(cfm_cfg.world_model), manifest["cfm"]["world_checkpoint"]),
        ("Drifting V7", OccupancyDriftGeneratorV3(drift_cfg), manifest["drifting"]["checkpoint"]),
        ("Drifting world (V6 origin)", V4BeliefLearningSystem(drift_cfg, controller_cfg.world_model), manifest["drifting"]["world_checkpoint"]),
    ]
    for name, model, checkpoint in models:
        load_checkpoint(ROOT / checkpoint, model, strict=True, map_location="cpu")
        model.eval()
        print(f"Strict load OK: {name}")
    if forward:
        with torch.inference_mode():
            c = cfm_cfg.cfm
            pose = torch.tensor([[0., 0., .2, 1., 0., 0., 0.]])
            condition = CFMCondition(
                image=torch.zeros(1,c.image_channels,c.image_size,c.image_size), pose=pose,
                observed_points=torch.rand(1,c.observed_points,3),
                observed_mask=torch.ones(1,c.observed_points,dtype=torch.bool),
                previous_points=torch.rand(1,c.num_points,3), previous_velocity=torch.zeros(1,c.num_points,3),
                confidence=torch.ones(1,c.num_points,1), particle_type=torch.zeros(1,c.num_points,1),
            )
            result = models[0][1](condition, num_steps=1)
            assert result.position.shape == (1,c.num_points,3) and torch.isfinite(result.position).all()
            from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
            f, size = drift_cfg.sequence_length, drift_cfg.image_size
            batch = {
                "rgb":torch.rand(1,f,3,size,size),
                "depth":torch.full((1,f,1,size,size),.8),
                "pose":pose[:,None].expand(1,f,7).clone(),
                "intrinsics":torch.tensor([[[90.,0,(size-1)/2],[0,90.,(size-1)/2],[0,0,1]]]),
            }
            result = reconstruct_occupancy_v3(models[2][1],batch,drift_cfg,ensemble_samples=1)
            assert torch.isfinite(result.position).all()
            print("Real-checkpoint CPU forward OK: CFM and Drifting (synthetic inputs, not performance evaluation)")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--forward",action="store_true",help="also run synthetic CPU inference with retained weights")
    args=parser.parse_args()
    manifest,count=validate_layout()
    check_models(manifest,args.forward)
    print(f"Unified layout OK; {count} checkpoint SHA-256 hashes unchanged; CPU only.")


if __name__ == "__main__":
    main()
