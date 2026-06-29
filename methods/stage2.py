from __future__ import annotations

import json
from collections import defaultdict
import time

from tqdm.auto import tqdm

from core.candidates import (
    EmotionWindowCandidate,
    build_emotion_windows,
    render_bridgeprot_prediction_json,
)
from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtMethodConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    BridgeProtRetrievalConfig,
    bridgeprot_execution_mode,
    resolve_bridgeprot_stage_methods,
)
from core.modalities import load_bridgeprot_multimodal_split
from core.prompts import SYSTEM_PROMPT, build_bridge_window_prompt
from core.runner import (
    BridgeProtResult,
    collect_bridgeprot_result,
    dialogue_results_to_pair_sets,
    run_bridgeprot_from_messages,
)
from core.serializer import serialize_utterance_line
from core.video_packing import VideoPackingStats, prepare_generate_prompts_for_inference
from core.vllm_engine import (
    build_lora_request,
    build_vllm_engine,
    build_window_sampling_params,
)
from methods.staged import run_bridgeprot_stage1_method


def run_bridgeprot_stage2(
    *,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    method_config: BridgeProtMethodConfig,
    retrieval_config: BridgeProtRetrievalConfig,
    seed: int,
    model_config: BridgeProtModelConfig | None = None,
    stage1_model_config: BridgeProtModelConfig | None = None,
    stage2_model_config: BridgeProtModelConfig | None = None,
    stage1_method: str | None = None,
    stage2_method: str | None = None,
) -> BridgeProtResult:
    resolved_stage, resolved_stage1_method, resolved_stage2_method = resolve_bridgeprot_stage_methods(method_config)
    if resolved_stage != 2:
        raise ValueError(f"run_bridgeprot_stage2 expected stage=2, but resolved stage={resolved_stage}.")
    if stage1_method is not None:
        resolved_stage1_method = stage1_method.lower()
    if stage2_method is not None:
        resolved_stage2_method = stage2_method.lower()

    default_model_config = stage2_model_config or stage1_model_config or model_config
    if default_model_config is None:
        raise ValueError("run_bridgeprot_stage2 requires a model configuration.")
    stage1_model_config = stage1_model_config or model_config or default_model_config
    stage2_model_config = stage2_model_config or model_config or default_model_config

    loaded_split = load_bridgeprot_multimodal_split(
        data_config,
        max_dialogues=data_config.max_dialogues,
    )
    dialogues = loaded_split.dialogues

    stage1_request_batch_size = _resolve_request_batch_size(stage1_model_config)
    stage2_request_batch_size = _resolve_request_batch_size(stage2_model_config)
    reuse_single_engine = stage1_model_config == stage2_model_config

    llm = None
    runtime_info = None
    lora_request = None
    if reuse_single_engine:
        llm, runtime_info = build_vllm_engine(
            stage2_model_config,
            seed=seed,
            execution_mode="multimodal-video" if data_config.use_video else "text-only",
        )
        lora_request = build_lora_request(stage2_model_config)

    raw_outputs: list[str] = []
    total_windows = 0
    stage2_packing_stats = VideoPackingStats()
    total_started_at = time.perf_counter()
    with tqdm(
        total=max(len(dialogues), 1),
        desc="BridgeProt stage2",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        progress_bar.set_postfix_str("phase=stage1")
        stage1_started_at = time.perf_counter()
        stage1_result = run_bridgeprot_stage1_method(
            method_name=resolved_stage1_method,
            dialogues=dialogues,
            data_config=data_config,
            model_config=stage1_model_config,
            protocol_config=protocol_config,
            decode_config=decode_config,
            retrieval_config=retrieval_config,
            seed=seed,
            llm=llm,
            lora_request=lora_request,
            runtime_info=runtime_info,
            request_batch_size=stage1_request_batch_size,
            progress_callback=progress_bar.update,
        )
        stage1_duration_sec = time.perf_counter() - stage1_started_at

        if not reuse_single_engine:
            llm, runtime_info = build_vllm_engine(
                stage2_model_config,
                seed=seed,
                execution_mode="multimodal-video" if data_config.use_video else "text-only",
            )
            lora_request = build_lora_request(stage2_model_config)

        seed_pairs_by_dialogue = dialogue_results_to_pair_sets(
            stage1_result.dialogue_results,
            pair_source=method_config.stage2_seed_pair_source,
        )
        base_pairs_by_dialogue = dialogue_results_to_pair_sets(
            stage1_result.dialogue_results,
            pair_source="strict",
        )
        window_prediction_rows: list[dict[str, object]] = []

        window_groups: list[list[EmotionWindowCandidate]] = []
        for dialogue, seed_pairs in zip(dialogues, seed_pairs_by_dialogue):
            windows = build_emotion_windows(
                num_turns=dialogue.num_utterances,
                seed_pairs=seed_pairs,
                method_config=method_config,
                protocol_config=protocol_config,
            )
            window_groups.append(windows)
            total_windows += len(windows)

        progress_bar.total = len(dialogues) + total_windows
        progress_bar.refresh()
        progress_bar.set_postfix_str("phase=stage2")
        stage2_started_at = time.perf_counter()

        for dialogue, windows, base_pairs in zip(dialogues, window_groups, base_pairs_by_dialogue):
            stage2_by_emotion, raw_window_rows, window_packing_stats = _score_dialogue_windows(
                dialogue=dialogue,
                windows=windows,
                data_config=data_config,
                model_config=stage2_model_config,
                method_config=method_config,
                llm=llm,
                lora_request=lora_request,
                batch_size=stage2_request_batch_size,
                progress_bar=progress_bar,
            )
            stage2_packing_stats.merge(window_packing_stats)
            accepted_pairs, fused_window_rows = _fuse_stage2_with_stage1(
                dialogue=dialogue,
                stage1_pairs=base_pairs,
                stage2_by_emotion=stage2_by_emotion,
                raw_window_rows=raw_window_rows,
                method_config=method_config,
                max_stage2_causes_per_emotion=method_config.max_stage2_causes_per_emotion,
            )
            window_prediction_rows.extend(fused_window_rows)
            raw_outputs.append(
                render_bridgeprot_prediction_json(
                    accepted_pairs=accepted_pairs,
                    protocol=protocol_config,
                    output_mode=decode_config.output_mode,
                )
            )
        stage2_duration_sec = time.perf_counter() - stage2_started_at
    total_duration_sec = time.perf_counter() - total_started_at

    stage2_result = collect_bridgeprot_result(
        dialogues=dialogues,
        raw_outputs=raw_outputs,
        protocol_config=protocol_config,
        output_mode=decode_config.output_mode,
        runtime_info={
            "num_visible_gpus": runtime_info.num_visible_gpus,
            "tensor_parallel_size": runtime_info.tensor_parallel_size,
            "gpu_memory_utilization": runtime_info.gpu_memory_utilization,
            "min_free_gpu_memory_gb": runtime_info.min_free_gpu_memory_gb,
            "num_dialogues": len(dialogues),
            "num_windows": total_windows,
            "stage": 2,
            "candidate_window": method_config.candidate_window,
            "stage2_mode": method_config.stage2_mode,
            "stage2_seed_pair_source": method_config.stage2_seed_pair_source,
            "max_stage2_causes_per_emotion": method_config.max_stage2_causes_per_emotion,
            "stage2_disallow_future_causes": int(method_config.stage2_disallow_future_causes),
            "stage2_drop_base_future_without_stage2_support": int(
                method_config.stage2_drop_base_future_without_stage2_support
            ),
            "stage2_drop_base_when_emotion_unsupported": int(
                method_config.stage2_drop_base_when_emotion_unsupported
            ),
            "stage2_replace_mode": method_config.stage2_replace_mode,
            "stage2_replace_max_causes_per_emotion": method_config.stage2_replace_max_causes_per_emotion,
            "stage1_method": resolved_stage1_method,
            "stage2_method": resolved_stage2_method,
            "stage1_strict_pair_f1": stage1_result.summary.strict_pair_f1,
            "stage1_salvaged_pair_f1": stage1_result.summary.salvaged_pair_f1,
            "stage1_strict_emotion_turn_f1": stage1_result.summary.strict_emotion_turn_f1,
            "stage1_salvaged_emotion_turn_f1": stage1_result.summary.salvaged_emotion_turn_f1,
            "stage1_strict_cause_turn_f1": stage1_result.summary.strict_cause_turn_f1,
            "stage1_salvaged_cause_turn_f1": stage1_result.summary.salvaged_cause_turn_f1,
            "stage2_video_messages": stage2_packing_stats.video_messages,
            "stage2_video_items_total": stage2_packing_stats.video_items_total,
            "stage2_video_items_kept": stage2_packing_stats.video_items_kept,
            "stage2_packed_messages_default_frames": stage2_packing_stats.packed_messages_default_frames,
            "stage2_packed_messages_fallback_frames": stage2_packing_stats.packed_messages_fallback_frames,
            "stage2_packed_messages_pruned": stage2_packing_stats.packed_messages_pruned,
            "stage2_packed_messages_dropped_all_video": stage2_packing_stats.packed_messages_dropped_all_video,
            "total_duration_sec": total_duration_sec,
            "stage1_duration_sec": stage1_duration_sec,
            "stage2_duration_sec": stage2_duration_sec,
            "stage1_dialogues_per_sec": (len(dialogues) / stage1_duration_sec) if stage1_duration_sec > 0 else 0.0,
            "stage2_windows_per_sec": (total_windows / stage2_duration_sec) if stage2_duration_sec > 0 else 0.0,
        },
    )
    stage2_result.stage_results["stage1"] = stage1_result
    stage2_result.artifacts["window_predictions.jsonl"] = window_prediction_rows
    return stage2_result

def _score_dialogue_windows(
    *,
    dialogue,
    windows: list[EmotionWindowCandidate],
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    method_config: BridgeProtMethodConfig,
    llm,
    lora_request,
    batch_size: int,
    progress_bar,
) -> tuple[dict[int, set[int]], list[dict[str, object]], VideoPackingStats]:
    if not windows:
        return {}, [], VideoPackingStats()

    max_items = max(1, (method_config.candidate_window * 2) + 1)
    sampling_params = build_window_sampling_params(max_items=max_items)
    proposed_by_emotion: dict[int, set[int]] = {}
    raw_rows: list[dict[str, object]] = []
    aggregate_packing_stats = VideoPackingStats()
    for window_batch in _iter_batches(windows, batch_size):
        messages = [
            _build_window_messages(
                dialogue=dialogue,
                window=window,
                data_config=data_config,
            )
            for window in window_batch
        ]
        if data_config.use_video:
            packed_messages, packing_stats = prepare_generate_prompts_for_inference(
                messages=messages,
                model_config=model_config,
                enable_budget_fallback=True,
            )
            outputs = llm.generate(
                packed_messages,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
        else:
            packed_messages = messages
            packing_stats = VideoPackingStats()
            outputs = llm.chat(
                packed_messages,
                sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
                chat_template_content_format="openai",
            )
        aggregate_packing_stats.merge(packing_stats)
        for window, output in zip(window_batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            emotion_supported, selected_turns = _parse_window_selection(
                raw_text,
                allowed_turns=set(window.candidate_turns),
            )
            proposed_by_emotion[int(window.emotion_turn)] = {
                int(turn) for turn in selected_turns
            }
            raw_rows.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "emotion_turn": int(window.emotion_turn),
                    "candidate_turns": [int(turn) for turn in window.candidate_turns],
                    "context_turns": [int(turn) for turn in window.context_turns],
                    "seeded_cause_turns": [int(turn) for turn in window.seeded_cause_turns],
                    "emotion_supported": bool(emotion_supported),
                    "raw_stage2_output": raw_text,
                    "stage2_selected_causes": [int(turn) for turn in selected_turns],
                }
            )
        progress_bar.update(len(window_batch))
    return proposed_by_emotion, raw_rows, aggregate_packing_stats


def _resolve_request_batch_size(model_config: BridgeProtModelConfig) -> int:
    if model_config.max_num_seqs is not None and model_config.max_num_seqs > 0:
        return model_config.max_num_seqs
    return 32


def _fuse_stage2_with_stage1(
    *,
    dialogue,
    stage1_pairs: set[tuple[int, int]],
    stage2_by_emotion: dict[int, set[int]],
    raw_window_rows: list[dict[str, object]],
    method_config: BridgeProtMethodConfig,
    max_stage2_causes_per_emotion: int,
) -> tuple[set[tuple[int, int]], list[dict[str, object]]]:
    stage1_by_emotion: dict[int, set[int]] = {}
    for emotion_turn, cause_turn in stage1_pairs:
        stage1_by_emotion.setdefault(int(emotion_turn), set()).add(int(cause_turn))

    raw_rows_by_emotion = {
        int(row["emotion_turn"]): row
        for row in raw_window_rows
    }
    gold_by_emotion: dict[int, set[int]] = defaultdict(set)
    for emotion_turn, cause_turn in dialogue.emotion_cause_pairs:
        gold_by_emotion[int(emotion_turn)].add(int(cause_turn))

    fused: set[tuple[int, int]] = set()
    fused_rows: list[dict[str, object]] = []
    for emotion_turn in sorted(set(stage1_by_emotion) | set(stage2_by_emotion)):
        base_causes = set(stage1_by_emotion.get(emotion_turn, set()))
        proposed_causes = set(stage2_by_emotion.get(emotion_turn, set()))
        raw_row = raw_rows_by_emotion.get(
            emotion_turn,
            {
                "candidate_turns": [],
                "context_turns": [],
                "seeded_cause_turns": [],
                "raw_stage2_output": "",
                "stage2_selected_causes": [],
            },
        )

        if method_config.stage2_disallow_future_causes:
            proposed_causes = {turn for turn in proposed_causes if turn <= emotion_turn}

        filtered_base_causes = set(base_causes)
        if method_config.stage2_drop_base_future_without_stage2_support:
            filtered_base_causes = {
                turn for turn in filtered_base_causes if turn <= emotion_turn or turn in proposed_causes
            }
        emotion_supported = bool(raw_row.get("emotion_supported", True))
        if method_config.stage2_drop_base_when_emotion_unsupported and not emotion_supported:
            filtered_base_causes = set()
            proposed_causes = set()

        replace_mode = method_config.stage2_replace_mode.lower()
        should_replace = False
        if proposed_causes:
            if replace_mode == "compact":
                should_replace = len(proposed_causes) <= method_config.stage2_replace_max_causes_per_emotion
            elif replace_mode == "subset":
                should_replace = proposed_causes.issubset(filtered_base_causes)
            elif replace_mode == "subset_or_compact":
                should_replace = proposed_causes.issubset(filtered_base_causes) or (
                    len(proposed_causes) <= method_config.stage2_replace_max_causes_per_emotion
                )

        if len(proposed_causes) <= max_stage2_causes_per_emotion:
            if should_replace:
                final_causes = proposed_causes
                decision = "replace"
            else:
                final_causes = filtered_base_causes | proposed_causes
                decision = "union"
        else:
            final_causes = filtered_base_causes
            decision = "fallback_base"
        for cause_turn in sorted(final_causes):
            fused.add((emotion_turn, cause_turn))
        fused_rows.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "emotion_turn": int(emotion_turn),
                "gold_causes": sorted(int(turn) for turn in gold_by_emotion.get(emotion_turn, set())),
                "stage1_base_causes": sorted(int(turn) for turn in base_causes),
                "stage1_base_causes_filtered": sorted(int(turn) for turn in filtered_base_causes),
                "stage2_proposed_causes": sorted(int(turn) for turn in proposed_causes),
                "final_causes": sorted(int(turn) for turn in final_causes),
                "candidate_turns": raw_row["candidate_turns"],
                "context_turns": raw_row["context_turns"],
                "seeded_cause_turns": raw_row["seeded_cause_turns"],
                "emotion_supported": emotion_supported,
                "raw_stage2_output": raw_row["raw_stage2_output"],
                "decision": decision,
            }
        )
    return fused, fused_rows
def _iter_batches(items, batch_size: int):
    effective_batch_size = max(1, batch_size)
    for start in range(0, len(items), effective_batch_size):
        yield items[start : start + effective_batch_size]


def _build_window_messages(
    *,
    dialogue,
    window: EmotionWindowCandidate,
    data_config: BridgeProtDataConfig,
) -> list[dict[str, object]]:
    serialized_context = _serialize_window_context(
        dialogue=dialogue,
        context_turns=window.context_turns,
        data_config=data_config,
    )
    audio_lines = _collect_audio_summary_lines(
        dialogue=dialogue,
        context_turns=window.context_turns,
        enabled=data_config.audio_summary_enabled,
    )
    prompt_text = build_bridge_window_prompt(
        serialized_context=serialized_context,
        emotion_turn=window.emotion_turn,
        candidate_turns=window.candidate_turns,
        seeded_cause_turns=window.seeded_cause_turns,
        audio_lines=audio_lines,
    )
    content: list[dict[str, object]] = [{"type": "text", "text": prompt_text}]
    _append_native_video_content(
        content=content,
        dialogue=dialogue,
        context_turns=window.context_turns,
        use_video=data_config.use_video,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _serialize_window_context(
    *,
    dialogue,
    context_turns: list[int],
    data_config: BridgeProtDataConfig,
) -> str:
    lines: list[str] = []
    for turn in context_turns:
        utterance_index = turn - 1
        if not (0 <= utterance_index < dialogue.num_utterances):
            continue
        lines.append(
            serialize_utterance_line(
                dialogue,
                utterance_index,
                include_turn_id=data_config.include_turn_id,
                include_speaker=data_config.include_speaker,
            )
        )
    return "\n".join(lines)


def _serialize_full_dialogue(
    *,
    dialogue,
    data_config: BridgeProtDataConfig,
) -> str:
    lines: list[str] = []
    for utterance_index in range(dialogue.num_utterances):
        lines.append(
            serialize_utterance_line(
                dialogue,
                utterance_index,
                include_turn_id=data_config.include_turn_id,
                include_speaker=data_config.include_speaker,
            )
        )
    return "\n".join(lines)


def _collect_audio_summary_lines(
    *,
    dialogue,
    context_turns: list[int],
    enabled: bool,
) -> list[str]:
    if not enabled:
        return []
    audio_lines: list[str] = []
    for turn in context_turns:
        utterance = dialogue.utterances[turn - 1]
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            audio_lines.append(f"Turn {turn}: {audio_summary}")
    return audio_lines


def _collect_dialogue_audio_summary_lines(dialogue) -> list[str]:
    audio_lines: list[str] = []
    for utterance in dialogue.utterances:
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            audio_lines.append(f"Turn {utterance.turn}: {audio_summary}")
    return audio_lines
def _append_native_video_content(
    *,
    content: list[dict[str, object]],
    dialogue,
    context_turns: list[int],
    use_video: bool,
) -> None:
    if not use_video:
        return

    seen_video_urls: set[str] = set()
    appended_native_video = False
    for turn in context_turns:
        utterance = dialogue.utterances[turn - 1]
        video_url = utterance.metadata.get("video_url")
        if not video_url or video_url in seen_video_urls:
            continue
        content.append({"type": "text", "text": f"[Window Turn Video] turn={turn}"})
        content.append({"type": "video_url", "video_url": {"url": video_url}})
        seen_video_urls.add(str(video_url))
        appended_native_video = True

    if appended_native_video:
        return

    dialogue_video = dialogue.metadata.get("video_url")
    if dialogue_video:
        content.append(
            {
                "type": "text",
                "text": "[Dialogue Video] The following clip covers the wider dialogue context for this local window.",
            }
        )
        content.append({"type": "video_url", "video_url": {"url": dialogue_video}})


def _parse_window_selection(
    raw_text: str,
    *,
    allowed_turns: set[int],
) -> tuple[bool, list[int]]:
    stripped = raw_text.strip()
    if not stripped:
        return False, []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return False, []
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return False, []
    if not isinstance(payload, dict):
        return False, []
    emotion_supported = payload.get("emotion_supported")
    if not isinstance(emotion_supported, bool):
        emotion_supported = bool(payload.get("cause_turns"))
    cause_turns = payload.get("cause_turns")
    if not isinstance(cause_turns, list):
        return bool(emotion_supported), []

    selected: set[int] = set()
    for item in cause_turns:
        try:
            turn = int(item)
        except (TypeError, ValueError):
            continue
        if turn in allowed_turns:
            selected.add(turn)
    if not emotion_supported:
        return False, []
    return True, sorted(selected)
