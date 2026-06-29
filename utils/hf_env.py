from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CACHE_ROOT = Path(os.environ.get("BRIDGEPROT_CACHE_ROOT", Path.home() / ".cache" / "bridgeprot")).expanduser()
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", DEFAULT_CACHE_ROOT / "hf")).expanduser()
DEFAULT_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", DEFAULT_CACHE_ROOT / "xdg")).expanduser()
DEFAULT_UNSLOTH_COMPILE_HOME = Path(
    os.environ.get("UNSLOTH_COMPILE_LOCATION", DEFAULT_CACHE_ROOT / "unsloth_compiled")
).expanduser()


def configure_hf_environment(hf_home: str | Path | None = None) -> dict[str, str]:
    resolved_hf_home = Path(hf_home) if hf_home is not None else DEFAULT_HF_HOME
    resolved_hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache = resolved_hf_home / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    resolved_xdg_cache = DEFAULT_XDG_CACHE_HOME
    resolved_xdg_cache.mkdir(parents=True, exist_ok=True)
    resolved_unsloth_compile = DEFAULT_UNSLOTH_COMPILE_HOME
    resolved_unsloth_compile.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(resolved_hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["XDG_CACHE_HOME"] = str(resolved_xdg_cache)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(resolved_unsloth_compile)
    os.environ.pop("TRANSFORMERS_CACHE", None)

    return {
        "HF_HOME": os.environ["HF_HOME"],
        "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
        "HF_HUB_DISABLE_PROGRESS_BARS": os.environ["HF_HUB_DISABLE_PROGRESS_BARS"],
        "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
        "UNSLOTH_COMPILE_LOCATION": os.environ["UNSLOTH_COMPILE_LOCATION"],
    }
