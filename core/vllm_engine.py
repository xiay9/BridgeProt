from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import warnings

from utils.hf_env import configure_hf_environment

configure_hf_environment()


def _append_pythonwarning_filter(filter_expr: str) -> None:
    raw_value = os.environ.get("PYTHONWARNINGS", "").strip()
    filters = [item for item in raw_value.split(",") if item]
    if filter_expr not in filters:
        filters.append(filter_expr)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def _configure_warning_filters() -> None:
    warning_specs = [
        ("ignore", r"To copy construct from a tensor.*", UserWarning),
        ("ignore", r"Input tensor shape suggests potential format mismatch.*", UserWarning),
    ]
    for action, message, category in warning_specs:
        warnings.filterwarnings(action, message=message, category=category)
        _append_pythonwarning_filter(f"{action}:{message}:{category.__name__}")


_configure_warning_filters()

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from core.config import (
    BridgeProtDecodeConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
)
from core.model_views import resolve_model_source_dir
from core.schema import build_bridgeprot_json_schema
from core.schema import build_emotion_window_selection_json_schema
from core.tokenizer_runtime import materialize_runtime_processor_view


@dataclass(slots=True)
class VLLMRuntimeInfo:
    num_visible_gpus: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    min_free_gpu_memory_gb: float


@dataclass(slots=True)
class VisibleGPUInfo:
    physical_index: int
    free_gb: float
    total_gb: float

    @property
    def free_fraction(self) -> float:
        return 0.0 if self.total_gb <= 0 else self.free_gb / self.total_gb


def _visible_physical_gpu_ids() -> list[int] | None:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return None

    visible_ids: list[int] = []
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        visible_ids.append(int(token))
    return visible_ids or None


def _query_visible_gpu_info() -> list[VisibleGPUInfo]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    all_rows: dict[int, VisibleGPUInfo] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            physical_index = int(parts[0])
            free_gb = float(parts[1]) / 1024.0
            total_gb = float(parts[2]) / 1024.0
        except ValueError:
            continue
        all_rows[physical_index] = VisibleGPUInfo(
            physical_index=physical_index,
            free_gb=free_gb,
            total_gb=total_gb,
        )

    visible_ids = _visible_physical_gpu_ids()
    if visible_ids is None:
        return [all_rows[index] for index in sorted(all_rows)]
    return [all_rows[index] for index in visible_ids if index in all_rows]


def detect_visible_gpu_count() -> int:
    visible_gpu_info = _query_visible_gpu_info()
    if visible_gpu_info:
        return len(visible_gpu_info)
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def resolve_tensor_parallel_size(requested: int) -> int:
    available = detect_visible_gpu_count()
    if available <= 0:
        raise RuntimeError("BridgeProt with vLLM requires at least one visible CUDA GPU.")
    if requested <= 0:
        return available
    if requested > available:
        raise ValueError(
            f"Requested tensor_parallel_size={requested}, but only {available} visible GPUs are available."
        )
    return requested


def resolve_gpu_memory_utilization(
    requested: float,
    *,
    tensor_parallel_size: int,
) -> tuple[float, float]:
    visible_gpu_info = _query_visible_gpu_info()
    if not visible_gpu_info:
        return requested, 0.0

    selected = visible_gpu_info[:tensor_parallel_size]
    if not selected:
        return requested, 0.0

    min_free_fraction = min(item.free_fraction for item in selected)
    min_free_gb = min(item.free_gb for item in selected)
    safety_margin = 0.10
    effective = min(requested, round(max(min_free_fraction - safety_margin, 0.0), 2))

    if effective < 0.5:
        summary = ", ".join(
            f"cuda:{idx} free={item.free_gb:.2f}GiB/{item.total_gb:.2f}GiB"
            for idx, item in enumerate(selected)
        )
        raise RuntimeError(
            "Visible GPUs do not have enough free memory for BridgeProt vLLM startup. "
            f"Current visible GPUs: {summary}. "
            "Please free GPU memory or reduce the number of visible GPUs."
        )
    return effective, min_free_gb


def build_vllm_engine(
    config: BridgeProtModelConfig,
    *,
    seed: int,
    execution_mode: str,
) -> tuple[LLM, VLLMRuntimeInfo]:
    tensor_parallel_size = resolve_tensor_parallel_size(config.tensor_parallel_size)
    gpu_memory_utilization, min_free_gpu_memory_gb = resolve_gpu_memory_utilization(
        config.gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    resolved_model_name = str(resolve_model_source_dir(config.model_name))
    tokenizer_name = str(resolve_model_source_dir(config.resolved_tokenizer_name))
    if execution_mode == "multimodal-video":
        tokenizer_name = materialize_runtime_processor_view(tokenizer_name)
    llm_kwargs = {
        "model": resolved_model_name,
        "tokenizer": tokenizer_name,
        "trust_remote_code": config.trust_remote_code,
        "dtype": config.dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "enforce_eager": config.enforce_eager,
        "seed": seed,
    }
    if config.hf_config_path is not None:
        llm_kwargs["hf_config_path"] = config.hf_config_path
    if config.hf_overrides:
        llm_kwargs["hf_overrides"] = config.hf_overrides
    if execution_mode == "text-only":
        llm_kwargs["language_model_only"] = True
    elif execution_mode == "multimodal-video":
        llm_kwargs["allowed_local_media_path"] = config.allowed_local_media_path
    else:
        raise ValueError(f"Unsupported BridgeProt execution mode '{execution_mode}'.")
    if config.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = config.max_num_seqs
    if config.lora_adapter_path is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1

    llm = LLM(
        **llm_kwargs,
    )
    return llm, VLLMRuntimeInfo(
        num_visible_gpus=detect_visible_gpu_count(),
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        min_free_gpu_memory_gb=min_free_gpu_memory_gb,
    )


def build_lora_request(config: BridgeProtModelConfig):
    if config.lora_adapter_path is None:
        return None

    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name="bridgeprot_adapter",
        lora_int_id=1,
        lora_path=str(Path(config.lora_adapter_path).resolve()),
        base_model_name=config.model_name,
    )


def build_sampling_params(
    config: BridgeProtDecodeConfig,
    *,
    protocol: BridgeProtProtocolConfig,
) -> SamplingParams:
    structured_outputs = StructuredOutputsParams(
        json=build_bridgeprot_json_schema(
            protocol,
            output_mode=config.output_mode,
        ),
        disable_additional_properties=True,
    )
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=min(config.max_tokens, 512),
        repetition_penalty=config.repetition_penalty,
        structured_outputs=structured_outputs,
    )


def build_window_sampling_params(
    *,
    max_items: int,
    max_tokens: int = 160,
    field_name: str = "cause_turns",
) -> SamplingParams:
    structured_outputs = StructuredOutputsParams(
        json=build_emotion_window_selection_json_schema(
            max_items=max_items,
            field_name=field_name,
        ),
        disable_additional_properties=True,
    )
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        repetition_penalty=1.0,
        structured_outputs=structured_outputs,
    )
