from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from configs.paths import DATA_ROOT, OUTPUT_ROOT


DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT


def _sanitize_output_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_.")
    return normalized or "na"


def _strip_trailing_semantic_suffixes(value: str, suffixes: Sequence[str]) -> str:
    normalized = value.strip()
    for suffix in reversed(tuple(item for item in suffixes if item)):
        pattern = re.compile(rf"(?:[._-]+{re.escape(suffix)})$", flags=re.IGNORECASE)
        updated = pattern.sub("", normalized).strip()
        if updated:
            normalized = updated
    return normalized or value.strip()


def bridgeprot_execution_mode(*, use_video: bool) -> str:
    return "multimodal-video" if use_video else "text-only"


def bridgeprot_evidence_tag(
    *,
    use_audio: bool,
    use_video: bool,
    use_audio_summary: bool = False,
) -> str:
    has_audio_evidence = use_audio or use_audio_summary
    has_video_evidence = use_video
    if has_video_evidence and has_audio_evidence:
        return "video-audio"
    if has_video_evidence:
        return "video"
    if has_audio_evidence:
        return "text-audio"
    return "text"

def bridgeprot_summary_tag(*, use_audio_summary: bool) -> str:
    return "audio-summary" if use_audio_summary else "no-summary"


def bridgeprot_stage_tag(stage: int) -> str:
    if stage != 2:
        raise ValueError(f"BridgeProt stage must be 2, got {stage}.")
    return f"stage{stage}"


def resolve_bridgeprot_stage_methods(method_config) -> tuple[int, str, str]:
    stage = int(getattr(method_config, "stage", 2))
    configured_name = str(getattr(method_config, "name", "zeroshot") or "zeroshot").lower()
    default_method = configured_name

    stage1_method = str(getattr(method_config, "stage1_method", None) or default_method).lower()
    stage2_method = str(getattr(method_config, "stage2_method", None) or default_method).lower()

    supported_methods = {"zeroshot", "fewshot", "lora", "sft"}
    if stage != 2:
        raise ValueError(f"Unsupported BridgeProt stage '{stage}'. Expected 2.")
    if stage1_method not in supported_methods:
        raise ValueError(
            f"Unsupported BridgeProt stage1_method '{stage1_method}'. "
            f"Expected one of: {', '.join(sorted(supported_methods))}."
        )
    if stage2_method not in supported_methods:
        raise ValueError(
            f"Unsupported BridgeProt stage2_method '{stage2_method}'. "
            f"Expected one of: {', '.join(sorted(supported_methods))}."
        )
    if stage1_method != stage2_method:
        raise ValueError(
            "BridgeProt now supports self stage2 only. "
            f"Received stage1_method='{stage1_method}' and stage2_method='{stage2_method}'. "
            "Use the same method for both stages."
        )
    return stage, stage1_method, stage2_method


def _build_bridgeprot_inference_dir_parts(
    *,
    run_name: str,
    method_name: str,
    stage: int,
    split: str,
    output_mode: str,
    use_audio: bool,
    use_video: bool,
    use_audio_summary: bool,
    timestamp: str,
    candidate_window: int | None = None,
    stage1_method: str | None = None,
    stage2_method: str | None = None,
) -> list[str]:
    execution_mode = bridgeprot_execution_mode(use_video=use_video)
    evidence_tag = bridgeprot_evidence_tag(
        use_audio=use_audio,
        use_video=use_video,
        use_audio_summary=use_audio_summary,
    )
    summary_tag = bridgeprot_summary_tag(
        use_audio_summary=use_audio_summary,
    )
    stage_tag = bridgeprot_stage_tag(stage)
    display_run_name = _strip_trailing_semantic_suffixes(
        run_name,
        (method_name, output_mode),
    )
    leaf_parts = [_sanitize_output_component(display_run_name)]
    if stage == 2 and candidate_window is not None:
        leaf_parts.append(f"candidate-window-{candidate_window}")
    if stage1_method:
        leaf_parts.append(f"stage1-{_sanitize_output_component(stage1_method)}")
    if stage == 2 and stage2_method:
        leaf_parts.append(f"stage2-{_sanitize_output_component(stage2_method)}")
    leaf_parts.append(_sanitize_output_component(timestamp))
    parts = [
        _sanitize_output_component(stage_tag),
        _sanitize_output_component(output_mode),
        _sanitize_output_component(execution_mode),
        _sanitize_output_component(evidence_tag),
        _sanitize_output_component(summary_tag),
    ]
    if split.lower() != "test":
        parts.append(f"split-{_sanitize_output_component(split)}")
    parts.append("--".join(leaf_parts))
    return parts


def _build_bridgeprot_training_dir_parts(
    *,
    run_name: str,
    method_name: str,
    target_mode: str,
    use_audio: bool,
    use_video: bool,
    use_audio_summary: bool,
    timestamp: str,
) -> list[str]:
    execution_mode = bridgeprot_execution_mode(use_video=use_video)
    evidence_tag = bridgeprot_evidence_tag(
        use_audio=use_audio,
        use_video=use_video,
        use_audio_summary=use_audio_summary,
    )
    summary_tag = bridgeprot_summary_tag(
        use_audio_summary=use_audio_summary,
    )
    display_run_name = _strip_trailing_semantic_suffixes(
        run_name,
        (method_name, target_mode),
    )
    return [
        _sanitize_output_component(method_name),
        _sanitize_output_component(target_mode),
        _sanitize_output_component(execution_mode),
        _sanitize_output_component(evidence_tag),
        _sanitize_output_component(summary_tag),
        "--".join(
            [
                _sanitize_output_component(display_run_name),
                _sanitize_output_component(timestamp),
            ]
        ),
    ]


def default_bridgeprot_output_dir(
    dataset_name: str,
    run_name: str,
    timestamp: str,
    *,
    method_name: str,
    stage: int,
    split: str,
    output_mode: str,
    use_audio: bool,
    use_video: bool,
    use_audio_summary: bool,
    candidate_window: int | None = None,
    stage1_method: str | None = None,
    stage2_method: str | None = None,
) -> str:
    return str(
        Path(
            DEFAULT_OUTPUT_ROOT,
            "bridgeprot",
            dataset_name.lower(),
            *_build_bridgeprot_inference_dir_parts(
                run_name=run_name,
                method_name=method_name,
                stage=stage,
                split=split,
                output_mode=output_mode,
                use_audio=use_audio,
                use_video=use_video,
                use_audio_summary=use_audio_summary,
                timestamp=timestamp,
                candidate_window=candidate_window,
                stage1_method=stage1_method,
                stage2_method=stage2_method,
            ),
        )
    )


@dataclass(slots=True)
class BridgeProtRunConfig:
    name: str
    seed: int = 42
    output_dir: str | None = None
    wandb_enabled: bool = True
    wandb_project: str = "BridgeProt"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: Sequence[str] = ()

    def resolve_output_dir(
        self,
        dataset_name: str,
        timestamp: str,
        *,
        method_name: str,
        stage: int,
        split: str,
        output_mode: str,
        use_audio: bool,
        use_video: bool,
        use_audio_summary: bool,
        candidate_window: int | None = None,
        stage1_method: str | None = None,
        stage2_method: str | None = None,
    ) -> str:
        return self.output_dir or default_bridgeprot_output_dir(
            dataset_name,
            self.name,
            timestamp,
            method_name=method_name,
            stage=stage,
            split=split,
            output_mode=output_mode,
            use_audio=use_audio,
            use_video=use_video,
            use_audio_summary=use_audio_summary,
            candidate_window=candidate_window,
            stage1_method=stage1_method,
            stage2_method=stage2_method,
        )


@dataclass(slots=True)
class BridgeProtDataConfig:
    dataset_name: str
    split: str = "test"
    data_root: Path = DEFAULT_DATA_ROOT
    max_dialogues: int | None = None
    include_speaker: bool = True
    include_turn_id: bool = True
    use_audio: bool = False
    use_video: bool = False
    use_audio_summary: bool | None = None
    audio_summary_backend: str = "opensmile_egemaps"

    @property
    def audio_summary_enabled(self) -> bool:
        if self.use_audio_summary is None:
            return bool(self.use_audio)
        return bool(self.use_audio_summary)

@dataclass(slots=True)
class BridgeProtModelConfig:
    model_name: str
    tokenizer_name: str | None = None
    lora_adapter_path: str | None = None
    hf_config_path: str | None = None
    hf_overrides: dict[str, object] | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 0
    enforce_eager: bool = False
    max_num_seqs: int | None = None
    allowed_local_media_path: str = str(DEFAULT_DATA_ROOT)
    video_fps: float | None = 1.0
    video_num_frames: int = 4
    video_fallback_num_frames: tuple[int, ...] = (4, 2, 1)
    video_max_edge: int = 224

    @property
    def resolved_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.model_name


@dataclass(slots=True)
class BridgeProtProtocolConfig:
    max_records: int = 32
    max_evidence_per_record: int = 4
    max_bridge_chars: int = 256
    max_explanation_chars: int = 512
    require_explanation: bool = True
    enforce_temporal_precedence: bool = False


@dataclass(slots=True)
class BridgeProtDecodeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    repetition_penalty: float = 1.0
    output_mode: str = "minimal"


@dataclass(slots=True)
class BridgeProtMethodConfig:
    name: str = "zeroshot"
    stage: int = 2
    stage1_method: str | None = None
    stage2_method: str | None = None
    stage2_mode: str = "emotion_window"
    candidate_window: int = 6
    max_emotion_windows: int | None = None
    fallback_to_window_scan: bool = True
    max_context_turns: int = 12
    stage2_seed_pair_source: str = "salvaged"
    max_stage2_causes_per_emotion: int = 3
    stage2_disallow_future_causes: bool = False
    stage2_drop_base_future_without_stage2_support: bool = False
    stage2_drop_base_when_emotion_unsupported: bool = False
    stage2_replace_mode: str = "off"
    stage2_replace_max_causes_per_emotion: int = 2


@dataclass(slots=True)
class BridgeProtRetrievalConfig:
    strategy: str = "semantic"
    num_shots: int = 3
    bank_split: str = "train"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_bank_size: int | None = None
    seed: int = 42
