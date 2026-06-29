from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.config import BridgeProtDataConfig
from data.media import materialize_video_paths
from data.registry import load_split
from data.schema import LoadedSplit


def load_bridgeprot_text_split(
    data_config: BridgeProtDataConfig,
    split: str | None = None,
    max_dialogues: int | None = None,
) -> LoadedSplit:
    loaded_split = _load_bridgeprot_base_split(data_config, split=split)
    _limit_loaded_split_dialogues(loaded_split, max_dialogues=max_dialogues)
    if data_config.audio_summary_enabled:
        _attach_audio_summaries(loaded_split, data_config)
    return loaded_split


def load_bridgeprot_multimodal_split(
    data_config: BridgeProtDataConfig,
    split: str | None = None,
    max_dialogues: int | None = None,
) -> LoadedSplit:
    loaded_split = _load_bridgeprot_base_split(data_config, split=split)
    _limit_loaded_split_dialogues(loaded_split, max_dialogues=max_dialogues)
    if data_config.audio_summary_enabled:
        _attach_audio_summaries(loaded_split, data_config)
    if data_config.use_video:
        _attach_standard_video_urls(loaded_split)
    return loaded_split


def _load_bridgeprot_base_split(
    data_config: BridgeProtDataConfig,
    split: str | None = None,
) -> LoadedSplit:
    resolved_split = split or data_config.split
    return load_split(
        data_config.dataset_name,
        resolved_split,
        data_root=data_config.data_root,
    )


def _limit_loaded_split_dialogues(loaded_split: LoadedSplit, *, max_dialogues: int | None) -> None:
    if max_dialogues is None:
        return
    loaded_split.dialogues = loaded_split.dialogues[:max_dialogues]


def _attach_audio_summaries(
    loaded_split: LoadedSplit,
    data_config: BridgeProtDataConfig,
) -> None:
    summary_map = _load_audio_summary_jsonl(
        dataset_root=loaded_split.root,
        split=loaded_split.split,
        backend_name=data_config.audio_summary_backend,
    )
    for dialogue in loaded_split.dialogues:
        for utterance in dialogue.utterances:
            summary_payload = summary_map.get((str(dialogue.dialogue_id), int(utterance.turn)))
            if not summary_payload:
                continue
            utterance.metadata["audio_summary"] = _format_audio_summary(summary_payload)


@lru_cache(maxsize=64)
def _load_audio_summary_jsonl(
    *,
    dataset_root: Path,
    split: str,
    backend_name: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    path = dataset_root / "cache" / backend_name / f"audio_summary_{split}.jsonl"
    if not path.exists():
        return {}

    summary_map: dict[tuple[str, int], dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            dialogue_id = row.get("dialogue_id")
            turn_id = row.get("turn_id")
            if dialogue_id is None or turn_id is None:
                continue
            payload = _coerce_audio_summary_payload(row.get("audio_summary"))
            if not payload:
                continue
            summary_map[(str(dialogue_id), int(turn_id))] = payload
    return summary_map


def _coerce_summary_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw_summary": stripped}
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_audio_summary_payload(value: Any) -> dict[str, Any] | None:
    return _coerce_summary_payload(value)


def _format_audio_summary(summary_payload: dict[str, Any]) -> str:
    return _format_summary_payload(summary_payload)


def _format_summary_payload(summary_payload: dict[str, Any]) -> str:
    segments: list[str] = []
    for key, value in summary_payload.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            rendered = ", ".join(str(item) for item in value if item is not None and str(item) != "")
            if not rendered:
                continue
            segments.append(f"{key}={rendered}")
            continue
        if isinstance(value, str) and not value.strip():
            continue
        segments.append(f"{key}={value}")
    return ", ".join(segments)

def _attach_standard_video_urls(loaded_split: LoadedSplit) -> None:
    materialize_video_paths(loaded_split)
    for dialogue in loaded_split.dialogues:
        dialogue_video_path = dialogue.metadata.get("video_path")
        if dialogue_video_path:
            dialogue.metadata["video_url"] = Path(str(dialogue_video_path)).resolve().as_uri()
        for utterance in dialogue.utterances:
            utterance_video_path = utterance.metadata.get("video_path")
            if utterance_video_path:
                utterance.metadata["video_url"] = Path(str(utterance_video_path)).resolve().as_uri()
