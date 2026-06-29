from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SUPPORTED_BRIDGEPROT_DATASETS = {
    "ecf": "ECF",
    "mec4": "MEC4",
    "mecad": "MECAD",
}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_input_path(path: str | Path, *, cwd: Path, code_root: Path, repo_root: Path) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(cwd / raw_path)
        candidates.append(code_root / raw_path)
        candidates.append(repo_root / raw_path)
        if raw_path.parts and raw_path.parts[0] == "code":
            stripped = Path(*raw_path.parts[1:])
            candidates.append(code_root / stripped)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find '{path}'.")


def load_yaml_config(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {resolved_path} must define a mapping.")
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_payload = load_yaml_config((resolved_path.parent / base_ref).resolve())
    return deep_merge_dicts(base_payload, payload)


def bridgeprot_infer_config_path(
    *,
    dataset: str,
    configs_root: Path,
) -> Path:
    dataset_key = dataset.lower()
    if dataset_key not in SUPPORTED_BRIDGEPROT_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_BRIDGEPROT_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of: {supported}")
    config_path = configs_root / "inference" / f"{dataset_key}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find inference config: {config_path}")
    return config_path


def bridgeprot_train_config_path(
    *,
    dataset: str,
    method: str,
    configs_root: Path,
) -> Path:
    dataset_key = dataset.lower()
    if dataset_key not in SUPPORTED_BRIDGEPROT_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_BRIDGEPROT_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of: {supported}")
    config_path = configs_root / "train" / f"{dataset_key}_{method.lower()}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find training config: {config_path}")
    return config_path
