from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from configs.paths import WANDB_ROOT


def ensure_wandb_dirs() -> dict[str, str]:
    paths = {
        "WANDB_DIR": WANDB_ROOT / "runs",
        "WANDB_CACHE_DIR": WANDB_ROOT / "cache",
        "WANDB_CONFIG_DIR": WANDB_ROOT / "config",
        "WANDB_DATA_DIR": WANDB_ROOT / "data",
    }
    resolved: dict[str, str] = {}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
        resolved[key] = os.environ[key]
    return resolved


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    group: str | None,
    tags: list[str],
    run_name: str,
    run_timestamp: str,
    output_dir: Path,
    config: dict[str, Any],
):
    if not enabled:
        return None

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B is enabled for this run, but importing wandb failed. "
            "Disable wandb or fix the local wandb installation."
        ) from exc

    ensure_wandb_dirs()
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            tags=tags,
            name=f"{run_name}-{run_timestamp}",
            mode="online",
            dir=str(output_dir),
            config=config,
            settings=wandb.Settings(max_end_of_run_history_metrics=0),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize W&B in online mode. "
            "Check wandb login or WANDB_API_KEY."
        ) from exc

    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("valid/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    run.define_metric("best/*", step_metric="epoch")
    return run
