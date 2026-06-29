from __future__ import annotations

import hashlib
import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp

from transformers import AutoProcessor


_RUNTIME_TOKENIZATION_KEYS = (
    "max_length",
    "truncation",
    "truncation_strategy",
    "stride",
    "pad_to_multiple_of",
)


def strip_runtime_tokenization_state(tokenizer_or_processor: object) -> None:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        if getattr(backend_tokenizer, "truncation", None) is not None:
            backend_tokenizer.no_truncation()
        if getattr(backend_tokenizer, "padding", None) is not None:
            backend_tokenizer.no_padding()

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for key in _RUNTIME_TOKENIZATION_KEYS:
            init_kwargs.pop(key, None)


@lru_cache(maxsize=16)
def materialize_runtime_processor_view(processor_name: str) -> str:
    source_dir = Path(processor_name).expanduser()
    if not source_dir.exists() or not source_dir.is_dir():
        return processor_name

    tokenizer_json = source_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        return processor_name

    try:
        tokenizer_payload = tokenizer_json.read_text(encoding="utf-8")
    except OSError:
        return processor_name

    # Training-time chat batching can persist tokenizer truncation into tokenizer.json.
    # For inference we need a clean processor view that does not inherit that runtime state.
    if '"truncation": {' not in tokenizer_payload and '"padding": {' not in tokenizer_payload:
        return processor_name

    cache_root = Path.home() / ".cache" / "pacever" / "bridgeprot" / "runtime_processors"
    cache_root.mkdir(parents=True, exist_ok=True)
    source_signature = (
        f"{source_dir.resolve()}::{tokenizer_json.stat().st_mtime_ns}::{tokenizer_json.stat().st_size}"
    )
    digest = hashlib.sha1(source_signature.encode("utf-8")).hexdigest()[:16]
    target_dir = cache_root / f"{source_dir.name}-{digest}"
    if target_dir.exists():
        return str(target_dir)

    tmp_dir = Path(mkdtemp(prefix=f".{target_dir.name}.", dir=str(cache_root)))
    processor = AutoProcessor.from_pretrained(str(source_dir), trust_remote_code=False)
    try:
        strip_runtime_tokenization_state(processor)
        processor.save_pretrained(tmp_dir)
        try:
            tmp_dir.rename(target_dir)
        except OSError:
            if target_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return str(target_dir)
            raise
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(target_dir)
