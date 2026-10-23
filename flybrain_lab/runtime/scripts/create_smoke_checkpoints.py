"""Create shape-valid random checkpoints only for environment integration tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_zbot.cfm import SecondOrderCFM
from codex_zbot.config import load_config
from codex_zbot.utils.checkpoint import save_checkpoint
from codex_zbot.utils.seed import seed_everything
from codex_zbot.world_model import BeliefLearningSystem


def main() -> None:
    cfg = load_config(ROOT / "configs" / "env_smoke.json")
    seed_everything(cfg.cfm.seed)
    output = ROOT / "checkpoints" / "env_smoke"
    save_checkpoint(output / "cfm.pt", SecondOrderCFM(cfg.cfm), None, 0, cfg.to_dict(), {"smoke_only": 1.0})
    save_checkpoint(
        output / "belief.pt", BeliefLearningSystem(cfg.world_model), None, 0, cfg.to_dict(), {"smoke_only": 1.0}
    )
    print(output)


if __name__ == "__main__":
    main()

