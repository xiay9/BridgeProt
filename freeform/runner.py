from __future__ import annotations

import time
from typing import Callable

from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    bridgeprot_execution_mode,
)
from core.modalities import load_bridgeprot_multimodal_split
from core.parse_back import dialogue_output_to_pair_set
from core.runner import (
    BridgeProtDialogueResult,
    BridgeProtResult,
    _chat_raw_outputs,
    _runtime_info_payload,
)
from core.schema import BridgeDialogueOutput
from core.serializer import serialize_dialogue
from core.metrics import summarize_bridgeprot
from freeform.parser import extract_freeform_records
from freeform.prompts import build_freeform_chat_messages


def build_freeform_sampling_params(config: BridgeProtDecodeConfig):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=min(config.max_tokens, 512),
        repetition_penalty=config.repetition_penalty,
    )


def build_freeform_messages(
    *,
    dialogues,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
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
            build_freeform_chat_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
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
def run_bridgeprot_freeform(
    *,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    seed: int,
    method_name: str = "zeroshot",
    retrieval_config=None,
    posthoc_extract: bool = False,
) -> BridgeProtResult:
    loaded_split = load_bridgeprot_multimodal_split(
        data_config,
        max_dialogues=data_config.max_dialogues,
    )
    dialogues = loaded_split.dialogues
    resolved_method = method_name.lower()
    if resolved_method in {"zeroshot", "lora", "sft"}:
        messages = build_freeform_messages(
            dialogues=dialogues,
            data_config=data_config,
            protocol_config=protocol_config,
        )
    elif resolved_method == "fewshot":
        if retrieval_config is None:
            raise ValueError("Free-form few-shot inference requires a retrieval_config.")
        from freeform.fewshot import build_freeform_fewshot_messages

        messages = build_freeform_fewshot_messages(
            dialogues=dialogues,
            data_config=data_config,
            model_config=model_config,
            protocol_config=protocol_config,
            decode_config=decode_config,
            retrieval_config=retrieval_config,
        )
    else:
        raise ValueError(
            f"Unsupported free-form method '{method_name}'. "
            "Expected one of: zeroshot, fewshot, lora, sft."
        )
    return run_bridgeprot_freeform_from_messages(
        dialogues=dialogues,
        messages=messages,
        model_config=model_config,
        protocol_config=protocol_config,
        decode_config=decode_config,
        seed=seed,
        execution_mode=bridgeprot_execution_mode(use_video=data_config.use_video),
        posthoc_extract=posthoc_extract,
    )


def run_bridgeprot_freeform_from_messages(
    *,
    dialogues,
    messages: list[list[dict[str, object]]],
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    seed: int,
    execution_mode: str,
    posthoc_extract: bool,
    llm=None,
    lora_request=None,
    runtime_info=None,
    request_batch_size: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> BridgeProtResult:
    from core.vllm_engine import build_lora_request, build_vllm_engine

    owns_engine = llm is None
    if owns_engine:
        llm, runtime_info = build_vllm_engine(
            model_config,
            seed=seed,
            execution_mode=execution_mode,
        )
    if lora_request is None:
        lora_request = build_lora_request(model_config)
    sampling_params = build_freeform_sampling_params(decode_config)
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
    return collect_bridgeprot_freeform_result(
        dialogues=dialogues,
        raw_outputs=raw_outputs,
        protocol_config=protocol_config,
        posthoc_extract=posthoc_extract,
        runtime_info=_runtime_info_payload(
            runtime_info=runtime_info,
            num_dialogues=len(dialogues),
            packing_stats=packing_stats,
            total_duration_sec=total_duration_sec,
        ),
    )


def collect_bridgeprot_freeform_result(
    *,
    dialogues,
    raw_outputs: list[str],
    protocol_config: BridgeProtProtocolConfig,
    posthoc_extract: bool,
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
        parse_result = extract_freeform_records(
            raw_text,
            num_turns=dialogue.num_utterances,
            protocol=protocol_config,
        )
        salvaged_output = BridgeDialogueOutput(records=parse_result.records)
        strict_valid = bool(posthoc_extract and parse_result.parsed and not parse_result.errors)
        strict_output = salvaged_output if strict_valid else None

        strict_pairs = (
            dialogue_output_to_pair_set(strict_output)
            if strict_output is not None
            else set()
        )
        salvaged_pairs = dialogue_output_to_pair_set(salvaged_output)
        gold_pairs = set(dialogue.emotion_cause_pairs)

        gold_pair_sets.append(gold_pairs)
        strict_pair_sets.append(strict_pairs)
        salvaged_pair_sets.append(salvaged_pairs)
        parsed_flags.append(parse_result.parsed)
        strict_valid_flags.append(strict_valid)
        total_records += max(parse_result.matched_record_count, len(parse_result.records))
        valid_records += len(parse_result.records)

        dialogue_results.append(
            BridgeProtDialogueResult(
                dataset=dialogue.dataset,
                split=dialogue.split,
                dialogue_id=dialogue.dialogue_id,
                raw_output=raw_text,
                parsed=parse_result.parsed,
                parse_strategy="freeform-posthoc" if posthoc_extract else "freeform-regex",
                strict_valid=strict_valid,
                errors=list(parse_result.errors),
                invalid_record_indices=list(parse_result.invalid_record_indices),
                gold_pairs=[list(pair) for pair in sorted(gold_pairs)],
                strict_pairs=[list(pair) for pair in sorted(strict_pairs)],
                salvaged_pairs=[list(pair) for pair in sorted(salvaged_pairs)],
                num_records=max(parse_result.matched_record_count, len(parse_result.records)),
                num_valid_records=len(parse_result.records),
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
