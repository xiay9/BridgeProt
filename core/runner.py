from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.video_packing import VideoPackingStats

from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    bridgeprot_execution_mode,
)
from core.modalities import load_bridgeprot_multimodal_split
from core.metrics import BridgeProtSummary, summarize_bridgeprot
from core.parse_back import dialogue_output_to_pair_set
from core.prompts import build_bridgeprot_chat_messages
from core.serializer import serialize_dialogue
from core.validator import BridgeValidationResult, validate_bridge_output


@dataclass(slots=True)
class BridgeProtDialogueResult:
    dataset: str
    split: str
    dialogue_id: str
    raw_output: str
    parsed: bool
    parse_strategy: str
    strict_valid: bool
    errors: list[str]
    invalid_record_indices: list[int]
    gold_pairs: list[list[int]]
    strict_pairs: list[list[int]]
    salvaged_pairs: list[list[int]]
    num_records: int
    num_valid_records: int


@dataclass(slots=True)
class BridgeProtResult:
    summary: BridgeProtSummary
    runtime_info: dict[str, int | str | float]
    dialogue_results: list[BridgeProtDialogueResult]
    stage_results: dict[str, "BridgeProtResult"] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)


def dialogue_results_to_pair_sets(
    dialogue_results: list[BridgeProtDialogueResult],
    *,
    pair_source: str = "salvaged",
) -> list[set[tuple[int, int]]]:
    attribute_name = {
        "gold": "gold_pairs",
        "strict": "strict_pairs",
        "salvaged": "salvaged_pairs",
    }.get(pair_source.lower())
    if attribute_name is None:
        raise ValueError(f"Unsupported pair source '{pair_source}'. Expected one of: gold, strict, salvaged.")

    pair_sets: list[set[tuple[int, int]]] = []
    for row in dialogue_results:
        pairs = getattr(row, attribute_name)
        pair_sets.append({(int(pair[0]), int(pair[1])) for pair in pairs})
    return pair_sets


def run_bridgeprot(
    *,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    seed: int,
) -> BridgeProtResult:
    loaded_split = load_bridgeprot_multimodal_split(
        data_config,
        max_dialogues=data_config.max_dialogues,
    )
    dialogues = loaded_split.dialogues

    messages = build_zeroshot_messages(
        dialogues=dialogues,
        data_config=data_config,
        protocol_config=protocol_config,
        output_mode=decode_config.output_mode,
    )

    return run_bridgeprot_from_messages(
        dialogues=dialogues,
        messages=messages,
        model_config=model_config,
        protocol_config=protocol_config,
        decode_config=decode_config,
        seed=seed,
        execution_mode=bridgeprot_execution_mode(use_video=data_config.use_video),
    )


def run_bridgeprot_from_messages(
    *,
    dialogues,
    messages: list[list[dict[str, object]]],
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    seed: int,
    execution_mode: str,
    llm=None,
    lora_request=None,
    runtime_info=None,
    request_batch_size: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> BridgeProtResult:
    from core.vllm_engine import build_lora_request, build_sampling_params, build_vllm_engine

    owns_engine = llm is None
    if owns_engine:
        llm, runtime_info = build_vllm_engine(
            model_config,
            seed=seed,
            execution_mode=execution_mode,
        )
    if lora_request is None:
        lora_request = build_lora_request(model_config)
    sampling_params = build_sampling_params(
        decode_config,
        protocol=protocol_config,
    )
    generation_started_at = time.perf_counter()
    raw_outputs, packing_stats = _chat_raw_outputs(
        llm=llm,
        messages=messages,
        sampling_params=sampling_params,
        lora_request=lora_request,
        model_config=model_config,
        execution_mode=execution_mode,
        request_batch_size=request_batch_size,
        progress_callback=progress_callback,
    )
    total_duration_sec = time.perf_counter() - generation_started_at
    return collect_bridgeprot_result(
        dialogues=dialogues,
        raw_outputs=raw_outputs,
        protocol_config=protocol_config,
        output_mode=decode_config.output_mode,
        runtime_info=_runtime_info_payload(
            runtime_info=runtime_info,
            num_dialogues=len(dialogues),
            packing_stats=packing_stats,
            total_duration_sec=total_duration_sec,
        ),
    )


def build_zeroshot_messages(
    *,
    dialogues,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
) -> list[list[dict[str, object]]]:
    conversations: list[list[dict[str, object]]] = []
    for dialogue in dialogues:
        serialized = serialize_dialogue(
            dialogue,
            include_turn_id=data_config.include_turn_id,
            include_speaker=data_config.include_speaker,
        )
        audio_lines = (
            _collect_dialogue_audio_summary_lines(dialogue)
            if data_config.audio_summary_enabled
            else []
        )
        conversations.append(
            build_bridgeprot_chat_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
                output_mode=output_mode,
                use_video=data_config.use_video,
                audio_lines=audio_lines,
            )
        )
    return conversations


def _collect_dialogue_audio_summary_lines(dialogue) -> list[str]:
    lines: list[str] = []
    for utterance in dialogue.utterances:
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            lines.append(f"Turn {utterance.turn}: {audio_summary}")
    return lines
def collect_bridgeprot_result(
    *,
    dialogues,
    raw_outputs: list[str],
    protocol_config: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
    runtime_info: dict[str, int | str | float],
) -> BridgeProtResult:
    gold_pair_sets: list[set[tuple[int, int]]] = []
    strict_pair_sets: list[set[tuple[int, int]]] = []
    salvaged_pair_sets: list[set[tuple[int, int]]] = []
    parsed_flags: list[bool] = []
    strict_valid_flags: list[bool] = []
    total_records = 0
    valid_records = 0
    dialogue_results: list[BridgeProtDialogueResult] = []

    for dialogue, raw_text in zip(dialogues, raw_outputs):
        validation = validate_bridge_output(
            raw_text,
            num_turns=dialogue.num_utterances,
            protocol=protocol_config,
            output_mode=output_mode,
        )
        strict_pairs = (
            dialogue_output_to_pair_set(validation.strict_output)
            if validation.strict_output is not None
            else set()
        )
        salvaged_pairs = dialogue_output_to_pair_set(validation.salvaged_output)
        gold_pairs = set(dialogue.emotion_cause_pairs)

        gold_pair_sets.append(gold_pairs)
        strict_pair_sets.append(strict_pairs)
        salvaged_pair_sets.append(salvaged_pairs)
        parsed_flags.append(validation.parsed)
        strict_valid_flags.append(validation.strict_valid)
        total_records += _num_total_records(validation)
        valid_records += len(validation.salvaged_output.records)

        dialogue_results.append(
            BridgeProtDialogueResult(
                dataset=dialogue.dataset,
                split=dialogue.split,
                dialogue_id=dialogue.dialogue_id,
                raw_output=raw_text,
                parsed=validation.parsed,
                parse_strategy=validation.parse_strategy,
                strict_valid=validation.strict_valid,
                errors=validation.errors,
                invalid_record_indices=validation.invalid_record_indices,
                gold_pairs=[list(pair) for pair in sorted(gold_pairs)],
                strict_pairs=[list(pair) for pair in sorted(strict_pairs)],
                salvaged_pairs=[list(pair) for pair in sorted(salvaged_pairs)],
                num_records=_num_total_records(validation),
                num_valid_records=len(validation.salvaged_output.records),
            )
        )

    summary = summarize_bridgeprot(
        gold_pair_sets=gold_pair_sets,
        strict_pair_sets=strict_pair_sets,
        salvaged_pair_sets=salvaged_pair_sets,
        parsed_flags=parsed_flags,
        strict_valid_flags=strict_valid_flags,
        total_records=total_records,
        valid_records=valid_records,
    )
    return BridgeProtResult(
        summary=summary,
        runtime_info=runtime_info,
        dialogue_results=dialogue_results,
    )


def _chat_raw_outputs(
    *,
    llm,
    messages: list[list[dict[str, object]]],
    sampling_params,
    lora_request,
    model_config: BridgeProtModelConfig,
    execution_mode: str,
    request_batch_size: int | None,
    progress_callback: Callable[[int], None] | None,
) -> tuple[list[str], VideoPackingStats | None]:
    if execution_mode == "multimodal-video":
        from core.video_packing import prepare_generate_prompts_for_inference

        packed_prompts, packing_stats = prepare_generate_prompts_for_inference(
            messages=messages,
            model_config=model_config,
            enable_budget_fallback=True,
            reserved_generation_tokens=1,
        )
        if request_batch_size is None or request_batch_size <= 0:
            outputs = llm.generate(
                packed_prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            if progress_callback is not None:
                progress_callback(len(packed_prompts))
            return [output.outputs[0].text if output.outputs else "" for output in outputs], packing_stats

        raw_outputs: list[str] = []
        for start in range(0, len(packed_prompts), request_batch_size):
            prompt_batch = packed_prompts[start : start + request_batch_size]
            outputs = llm.generate(
                prompt_batch,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            raw_outputs.extend(output.outputs[0].text if output.outputs else "" for output in outputs)
            if progress_callback is not None:
                progress_callback(len(prompt_batch))
        return raw_outputs, packing_stats

    packed_messages = messages
    if request_batch_size is None or request_batch_size <= 0:
        outputs = llm.chat(
            packed_messages,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
            chat_template_content_format="openai",
        )
        if progress_callback is not None:
            progress_callback(len(packed_messages))
        return [output.outputs[0].text if output.outputs else "" for output in outputs], None

    raw_outputs: list[str] = []
    for start in range(0, len(packed_messages), request_batch_size):
        message_batch = packed_messages[start : start + request_batch_size]
        outputs = llm.chat(
            message_batch,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
            chat_template_content_format="openai",
        )
        raw_outputs.extend(output.outputs[0].text if output.outputs else "" for output in outputs)
        if progress_callback is not None:
            progress_callback(len(message_batch))
    return raw_outputs, None


def _runtime_info_payload(
    *,
    runtime_info,
    num_dialogues: int,
    packing_stats: VideoPackingStats | None = None,
    total_duration_sec: float | None = None,
) -> dict[str, int | str | float]:
    if runtime_info is None:
        payload = {"num_dialogues": num_dialogues}
    else:
        payload = {
        "num_visible_gpus": runtime_info.num_visible_gpus,
        "tensor_parallel_size": runtime_info.tensor_parallel_size,
        "gpu_memory_utilization": runtime_info.gpu_memory_utilization,
        "min_free_gpu_memory_gb": runtime_info.min_free_gpu_memory_gb,
        "num_dialogues": num_dialogues,
    }
    if total_duration_sec is not None:
        payload.update(
            {
                "total_duration_sec": total_duration_sec,
                "dialogues_per_sec": (num_dialogues / total_duration_sec) if total_duration_sec > 0 else 0.0,
            }
        )
    if packing_stats is not None:
        payload.update(
            {
                "video_messages": packing_stats.video_messages,
                "video_items_total": packing_stats.video_items_total,
                "video_items_kept": packing_stats.video_items_kept,
                "packed_messages_default_frames": packing_stats.packed_messages_default_frames,
                "packed_messages_fallback_frames": packing_stats.packed_messages_fallback_frames,
                "packed_messages_pruned": packing_stats.packed_messages_pruned,
                "packed_messages_dropped_all_video": packing_stats.packed_messages_dropped_all_video,
            }
        )
    return payload


def _num_total_records(validation: BridgeValidationResult) -> int:
    if validation.strict_output is not None:
        return len(validation.strict_output.records)
    if validation.parsed:
        return len(validation.salvaged_output.records) + len(validation.invalid_record_indices)
    return 0
