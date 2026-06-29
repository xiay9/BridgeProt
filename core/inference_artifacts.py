from __future__ import annotations

import json
from pathlib import Path

from core.config import BridgeProtModelConfig


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def resolve_inference_model_config_from_training_output(
    *,
    train_output_dir: str | Path,
    base_model_config: BridgeProtModelConfig,
    use_video: bool,
    expected_method: str | None = None,
) -> BridgeProtModelConfig:
    train_dir = Path(train_output_dir).expanduser().resolve()
    train_config = _load_json(train_dir / "train_config.json")
    train_result = _load_json(train_dir / "train_result.json")

    training_section = train_config.get("training", {})
    data_section = train_config.get("data", {})
    training_method = str(training_section.get("method", "")).lower()
    if training_method not in {"lora", "sft"}:
        raise ValueError(
            f"Training output {train_dir} does not describe a supported fine-tuned method. "
            f"Expected lora or sft, got '{training_method or 'unknown'}'."
        )

    if expected_method is not None and training_method != expected_method.lower():
        raise ValueError(
            f"Training output {train_dir} was produced by method '{training_method}', "
            f"but '{expected_method}' was requested."
        )

    train_used_video = bool(data_section.get("use_video", False))
    best_model_checkpoint = train_result.get("best_model_checkpoint")
    merged_model_dir = train_result.get("merged_model_dir")
    text_model_view_dir = train_dir / "text_model_view"

    model_name = base_model_config.model_name
    tokenizer_name = base_model_config.tokenizer_name
    lora_adapter_path = None
    hf_config_path = base_model_config.hf_config_path
    hf_overrides = dict(base_model_config.hf_overrides or {})
    enforce_eager = base_model_config.enforce_eager
    max_num_seqs = base_model_config.max_num_seqs

    if training_method == "sft":
        if best_model_checkpoint:
            checkpoint_dir = Path(best_model_checkpoint).resolve()
            model_name = str(checkpoint_dir)
            tokenizer_name = str(checkpoint_dir)
        else:
            model_name = str(train_dir)
            tokenizer_name = str(train_dir)
        if not use_video:
            hf_config_path = base_model_config.model_name
            hf_overrides["architectures"] = ["Qwen3_5ForCausalLM"]
    else:
        if merged_model_dir:
            merged_dir = Path(merged_model_dir).resolve()
            model_name = str(merged_dir)
            tokenizer_name = str(merged_dir)
            if not use_video:
                hf_config_path = base_model_config.model_name
                hf_overrides["architectures"] = ["Qwen3_5ForCausalLM"]
        else:
            adapter_dir = Path(best_model_checkpoint).resolve() if best_model_checkpoint else train_dir
            if not use_video and text_model_view_dir.exists():
                model_name = str(text_model_view_dir.resolve())
                tokenizer_name = str(text_model_view_dir.resolve())
                hf_config_path = base_model_config.model_name
                hf_overrides["architectures"] = ["Qwen3_5ForCausalLM"]
            else:
                model_name = base_model_config.model_name
                tokenizer_name = str(train_dir)
            lora_adapter_path = str(adapter_dir)
            enforce_eager = True
            max_num_seqs = base_model_config.max_num_seqs or 8

    return BridgeProtModelConfig(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        lora_adapter_path=lora_adapter_path,
        hf_config_path=hf_config_path,
        hf_overrides=hf_overrides or None,
        trust_remote_code=base_model_config.trust_remote_code,
        dtype=base_model_config.dtype,
        max_model_len=base_model_config.max_model_len,
        gpu_memory_utilization=base_model_config.gpu_memory_utilization,
        tensor_parallel_size=base_model_config.tensor_parallel_size,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
        allowed_local_media_path=base_model_config.allowed_local_media_path,
        video_fps=base_model_config.video_fps,
        video_num_frames=base_model_config.video_num_frames,
        video_fallback_num_frames=base_model_config.video_fallback_num_frames,
        video_max_edge=base_model_config.video_max_edge,
    )
