from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoConfig


def materialize_text_only_model_view(
    *,
    model_name: str,
    trust_remote_code: bool,
    prefix: str,
) -> tuple[str, TemporaryDirectory | None]:
    source_dir = resolve_model_source_dir(model_name)
    model_config = AutoConfig.from_pretrained(
        str(source_dir),
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(model_config, "vision_config"):
        return str(source_dir), None

    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        raise ValueError(
            f"BridgeProt text-only execution expected a text sub-config for '{model_name}', but none was found."
        )

    text_config = copy.deepcopy(text_config)
    text_config.architectures = ["Qwen3_5ForCausalLM"]

    temp_view = TemporaryDirectory(prefix=prefix)
    temp_path = Path(temp_view.name)
    temp_path.mkdir(parents=True, exist_ok=True)

    for child in source_dir.iterdir():
        target = temp_path / child.name
        if child.name == "config.json":
            continue
        _symlink_or_copy(child, target)

    with open(temp_path / "config.json", "w", encoding="utf-8") as handle:
        json.dump(text_config.to_dict(), handle, indent=2, ensure_ascii=False)
    return str(temp_path), temp_view


def resolve_model_source_dir(
    model_name: str,
    *,
    max_network_retries: int = 3,
    initial_retry_delay_seconds: float = 2.0,
) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        return local_path.resolve()

    try:
        return Path(snapshot_download(model_name, local_files_only=True)).resolve()
    except (FileNotFoundError, LocalEntryNotFoundError):
        pass

    delay_seconds = initial_retry_delay_seconds
    last_error: Exception | None = None
    for attempt in range(1, max_network_retries + 1):
        try:
            return Path(snapshot_download(model_name)).resolve()
        except Exception as exc:
            last_error = exc
            if not _is_retryable_snapshot_error(exc) or attempt >= max_network_retries:
                raise
            time.sleep(delay_seconds)
            delay_seconds *= 2.0

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to resolve model source directory for '{model_name}'.")


def _is_retryable_snapshot_error(exc: Exception) -> bool:
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return False
        return status_code == 429 or status_code >= 500
    return False


def _symlink_or_copy(source: Path, target: Path) -> None:
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
        return
    except OSError:
        pass

    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target)
