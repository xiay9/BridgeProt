from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BRIDGEPROT_DATA_ROOT", REPO_ROOT / "datasets")).expanduser()
SCRATCH_ROOT = Path(os.environ.get("BRIDGEPROT_SCRATCH_ROOT", REPO_ROOT)).expanduser()
OUTPUT_ROOT = Path(os.environ.get("BRIDGEPROT_OUTPUT_ROOT", SCRATCH_ROOT / "outputs")).expanduser()
WANDB_ROOT = OUTPUT_ROOT / "wandb"
