from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

from core.config import BridgeProtModelConfig
from core.video_decode import sample_video_frames
from core.tokenizer_runtime import materialize_runtime_processor_view
from core.tokenizer_runtime import strip_runtime_tokenization_state


@dataclass(slots=True)
class PackedVideoClip:
    frames: np.ndarray
    metadata: dict[str, object]


@dataclass(slots=True)
class VideoPackingStats:
    video_messages: int = 0
    video_items_total: int = 0
    video_items_kept: int = 0
    packed_messages_default_frames: int = 0
    packed_messages_fallback_frames: int = 0
    packed_messages_pruned: int = 0
    packed_messages_dropped_all_video: int = 0

    def merge(self, other: "VideoPackingStats") -> None:
        self.video_messages += other.video_messages
        self.video_items_total += other.video_items_total
        self.video_items_kept += other.video_items_kept
        self.packed_messages_default_frames += other.packed_messages_default_frames
        self.packed_messages_fallback_frames += other.packed_messages_fallback_frames
        self.packed_messages_pruned += other.packed_messages_pruned
        self.packed_messages_dropped_all_video += other.packed_messages_dropped_all_video


@dataclass(slots=True)
class _VideoSlot:
    slot_id: int
    message_index: int
    content_index: int
    url: str
    is_query: bool


def prepare_generate_prompts_for_inference(
    *,
    messages: list[list[dict[str, object]]],
    model_config: BridgeProtModelConfig,
    enable_budget_fallback: bool,
    reserved_generation_tokens: int = 1,
) -> tuple[list[dict[str, object]], VideoPackingStats]:
    if not messages:
        return messages, VideoPackingStats()

    aggregate = VideoPackingStats()
    packed_batches: list[dict[str, object]] = []
    for message in messages:
        packed_message, stats = _prepare_single_generate_prompt(
            message=message,
            model_config=model_config,
            enable_budget_fallback=enable_budget_fallback,
            reserved_generation_tokens=reserved_generation_tokens,
        )
        packed_batches.append(packed_message)
        aggregate.merge(stats)
    return packed_batches, aggregate


def _prepare_single_generate_prompt(
    *,
    message: list[dict[str, object]],
    model_config: BridgeProtModelConfig,
    enable_budget_fallback: bool,
    reserved_generation_tokens: int,
) -> tuple[dict[str, object], VideoPackingStats]:
    slots = _collect_video_slots(message)
    processor_name = model_config.resolved_tokenizer_name
    if not slots:
        prompt_text = _render_chat_prompt(message, processor_name=processor_name)
        return {"prompt": prompt_text}, VideoPackingStats()

    stats = VideoPackingStats(video_messages=1, video_items_total=len(slots))
    frame_candidates = _resolve_frame_candidates(model_config)
    # vLLM generate requires the prompt to leave room for at least one output token.
    max_prompt_len = max(1, model_config.max_model_len - max(1, reserved_generation_tokens))

    for candidate_index, num_frames in enumerate(frame_candidates):
        packed_clips = {
            slot.slot_id: _pack_video_clip(
                url=slot.url,
                num_frames=num_frames,
                max_edge=model_config.video_max_edge,
            )
            for slot in slots
        }
        if not enable_budget_fallback:
            stats.video_items_kept = len(slots)
            if candidate_index == 0:
                stats.packed_messages_default_frames += 1
            else:
                stats.packed_messages_fallback_frames += 1
            return _build_generate_prompt(
                message=message,
                slots=slots,
                keep_slot_ids={slot.slot_id for slot in slots},
                packed_clips=packed_clips,
                processor_name=processor_name,
            ), stats

        estimated_tokens = _estimate_prompt_tokens(
            message=message,
            slots=slots,
            keep_slot_ids={slot.slot_id for slot in slots},
            packed_clips=packed_clips,
            processor_name=processor_name,
        )
        if estimated_tokens <= max_prompt_len:
            stats.video_items_kept = len(slots)
            if candidate_index == 0:
                stats.packed_messages_default_frames += 1
            else:
                stats.packed_messages_fallback_frames += 1
            return _build_generate_prompt(
                message=message,
                slots=slots,
                keep_slot_ids={slot.slot_id for slot in slots},
                packed_clips=packed_clips,
                processor_name=processor_name,
            ), stats

        if candidate_index != len(frame_candidates) - 1:
            continue

        keep_slot_ids = _find_best_pruned_selection(
            message=message,
            slots=slots,
            packed_clips=packed_clips,
            processor_name=processor_name,
            max_prompt_len=max_prompt_len,
        )
        stats.video_items_kept = len(keep_slot_ids)
        if keep_slot_ids == {slot.slot_id for slot in slots}:
            stats.packed_messages_default_frames += 1
        else:
            stats.packed_messages_fallback_frames += 1
            stats.packed_messages_pruned += 1
            if not keep_slot_ids:
                stats.packed_messages_dropped_all_video += 1
        return _build_generate_prompt(
            message=message,
            slots=slots,
            keep_slot_ids=keep_slot_ids,
            packed_clips=packed_clips,
            processor_name=processor_name,
        ), stats

    prompt_text = _render_chat_prompt(message, processor_name=processor_name)
    return {"prompt": prompt_text}, stats


def _resolve_frame_candidates(model_config: BridgeProtModelConfig) -> tuple[int, ...]:
    resolved: list[int] = []
    seen: set[int] = set()
    for value in (model_config.video_num_frames, *model_config.video_fallback_num_frames):
        frames = int(value)
        if frames <= 0 or frames in seen:
            continue
        resolved.append(frames)
        seen.add(frames)
    if not resolved:
        return (4, 2, 1)
    return tuple(resolved)


def _collect_video_slots(message: list[dict[str, object]]) -> list[_VideoSlot]:
    last_user_index = max(
        (index for index, msg in enumerate(message) if msg.get("role") == "user"),
        default=-1,
    )
    slots: list[_VideoSlot] = []
    slot_id = 0
    for message_index, msg in enumerate(message):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "video_url":
                continue
            video_payload = item.get("video_url")
            if not isinstance(video_payload, dict):
                continue
            url = video_payload.get("url")
            if not isinstance(url, str) or not url:
                continue
            slots.append(
                _VideoSlot(
                    slot_id=slot_id,
                    message_index=message_index,
                    content_index=content_index,
                    url=url,
                    is_query=message_index == last_user_index,
                )
            )
            slot_id += 1
    return slots


def _find_best_pruned_selection(
    *,
    message: list[dict[str, object]],
    slots: list[_VideoSlot],
    packed_clips: dict[int, PackedVideoClip],
    processor_name: str,
    max_prompt_len: int,
) -> set[int]:
    query_slots = [slot for slot in slots if slot.is_query]
    demo_slots = [slot for slot in slots if not slot.is_query]

    best_demo_keep = _max_keep_count_that_fits(
        message=message,
        fixed_slots=query_slots,
        candidate_slots=demo_slots,
        packed_clips=packed_clips,
        processor_name=processor_name,
        max_prompt_len=max_prompt_len,
    )
    if best_demo_keep is not None:
        selected = _slot_id_set(query_slots) | _slot_id_set(_uniform_take(demo_slots, best_demo_keep))
        return selected

    best_query_keep = _max_keep_count_that_fits(
        message=message,
        fixed_slots=[],
        candidate_slots=query_slots,
        packed_clips=packed_clips,
        processor_name=processor_name,
        max_prompt_len=max_prompt_len,
    )
    if best_query_keep is not None:
        return _slot_id_set(_uniform_take(query_slots, best_query_keep))
    return set()


def _max_keep_count_that_fits(
    *,
    message: list[dict[str, object]],
    fixed_slots: list[_VideoSlot],
    candidate_slots: list[_VideoSlot],
    packed_clips: dict[int, PackedVideoClip],
    processor_name: str,
    max_prompt_len: int,
) -> int | None:
    all_slots = fixed_slots + candidate_slots
    fixed_ids = _slot_id_set(fixed_slots)
    if _estimate_prompt_tokens(
        message=message,
        slots=all_slots,
        keep_slot_ids=fixed_ids,
        packed_clips=packed_clips,
        processor_name=processor_name,
    ) > max_prompt_len:
        return None

    low = 0
    high = len(candidate_slots)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        selected_ids = fixed_ids | _slot_id_set(_uniform_take(candidate_slots, mid))
        estimated_tokens = _estimate_prompt_tokens(
            message=message,
            slots=all_slots,
            keep_slot_ids=selected_ids,
            packed_clips=packed_clips,
            processor_name=processor_name,
        )
        if estimated_tokens <= max_prompt_len:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _estimate_prompt_tokens(
    *,
    message: list[dict[str, object]],
    slots: list[_VideoSlot],
    keep_slot_ids: set[int],
    packed_clips: dict[int, PackedVideoClip],
    processor_name: str,
) -> int:
    estimator_message = _materialize_message(
        message=message,
        slots=slots,
        keep_slot_ids=keep_slot_ids,
        packed_clips=packed_clips,
    )
    processor = _load_processor(processor_name)
    templated = _render_chat_prompt(estimator_message, processor_name=processor_name)
    kept_slots = [slot for slot in slots if slot.slot_id in keep_slot_ids]
    if not kept_slots:
        encoded = processor(text=[templated], return_tensors="pt")
        return int(encoded["input_ids"].shape[-1])

    videos = [[packed_clips[slot.slot_id].frames for slot in kept_slots]]
    metadata = [[_to_video_metadata(packed_clips[slot.slot_id].metadata) for slot in kept_slots]]
    encoded = processor(
        text=[templated],
        videos=videos,
        video_metadata=metadata,
        do_sample_frames=False,
        return_tensors="pt",
    )
    return int(encoded["input_ids"].shape[-1])


def _build_generate_prompt(
    *,
    message: list[dict[str, object]],
    slots: list[_VideoSlot],
    keep_slot_ids: set[int],
    packed_clips: dict[int, PackedVideoClip],
    processor_name: str,
) -> dict[str, object]:
    estimator_message = _materialize_message(
        message=message,
        slots=slots,
        keep_slot_ids=keep_slot_ids,
        packed_clips=packed_clips,
    )
    prompt_text = _render_chat_prompt(estimator_message, processor_name=processor_name)
    kept_slots = [slot for slot in slots if slot.slot_id in keep_slot_ids]
    if not kept_slots:
        return {"prompt": prompt_text}
    return {
        "prompt": prompt_text,
        "multi_modal_data": {
            "video": [
                (packed_clips[slot.slot_id].frames, packed_clips[slot.slot_id].metadata)
                for slot in kept_slots
            ]
        },
        "mm_processor_kwargs": {"do_sample_frames": False},
    }


def _materialize_message(
    *,
    message: list[dict[str, object]],
    slots: list[_VideoSlot],
    keep_slot_ids: set[int],
    packed_clips: dict[int, PackedVideoClip],
) -> list[dict[str, object]]:
    slot_lookup = {(slot.message_index, slot.content_index): slot for slot in slots}
    updated_message = deepcopy(message)
    for message_index, msg in enumerate(updated_message):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict[str, object]] = []
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            slot = slot_lookup.get((message_index, content_index))
            if slot is None:
                new_content.append(item)
                continue

            if slot.slot_id not in keep_slot_ids:
                if new_content and _is_video_label(new_content[-1]):
                    new_content.pop()
                continue

            new_content.append({"type": "video"})
        msg["content"] = new_content
    return updated_message


def _is_video_label(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") != "text":
        return False
    text = item.get("text")
    return isinstance(text, str) and "video" in text.lower()


def _uniform_take(slots: list[_VideoSlot], keep_count: int) -> list[_VideoSlot]:
    if keep_count <= 0:
        return []
    if keep_count >= len(slots):
        return list(slots)
    total = len(slots)
    indices = [int(position * total / keep_count) for position in range(keep_count)]
    return [slots[min(index, total - 1)] for index in indices]


def _slot_id_set(slots: list[_VideoSlot]) -> set[int]:
    return {slot.slot_id for slot in slots}


@lru_cache(maxsize=4)
def _load_processor(processor_name: str):
    runtime_processor_name = materialize_runtime_processor_view(processor_name)
    processor = AutoProcessor.from_pretrained(runtime_processor_name, trust_remote_code=False)
    strip_runtime_tokenization_state(processor)
    return processor


def _render_chat_prompt(message: list[dict[str, object]], *, processor_name: str) -> str:
    processor = _load_processor(processor_name)
    return processor.apply_chat_template(
        _normalize_message_for_processor_template(message),
        tokenize=False,
        add_generation_prompt=True,
    )


_TEMPLATE_PLACEHOLDERS = {
    "video": "[Video]",
    "video_url": "[Video]",
    "image": "[Image]",
    "image_url": "[Image]",
    "audio": "[Audio]",
    "audio_url": "[Audio]",
}


def _normalize_message_for_processor_template(
    message: list[dict[str, object]],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in message:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, list):
            normalized.append(
                {
                    "role": role,
                    "content": _normalize_content_list_for_processor_template(content),
                }
            )
            continue
        normalized.append(
            {
                "role": role,
                "content": "" if content is None else str(content),
            }
        )
    return normalized


def _normalize_content_list_for_processor_template(
    content: list[object],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in content:
        normalized_item = _normalize_content_item_for_processor_template(item)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _normalize_content_item_for_processor_template(
    item: object,
) -> dict[str, object] | None:
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        if item is None:
            return None
        return {"type": "text", "text": str(item)}

    item_type = str(item.get("type") or "").strip()
    if item_type == "text":
        text = item.get("text")
        return {"type": "text", "text": text if isinstance(text, str) else ""}
    if item_type == "video":
        return {"type": "video"}
    if item_type == "image":
        return {"type": "image"}
    if item_type in _TEMPLATE_PLACEHOLDERS:
        return {"type": "text", "text": _TEMPLATE_PLACEHOLDERS[item_type]}
    if item_type:
        return {"type": "text", "text": f"[{item_type}]"}

    text = item.get("text")
    if isinstance(text, str):
        return {"type": "text", "text": text}
    return None


@lru_cache(maxsize=256)
def _pack_video_clip(
    *,
    url: str,
    num_frames: int,
    max_edge: int,
) -> PackedVideoClip:
    path = _resolve_video_path(url)
    frames, metadata = sample_video_frames(path, num_frames=num_frames, max_edge=max_edge)
    return PackedVideoClip(
        frames=frames,
        metadata={
            "fps": metadata.get("fps"),
            "duration": metadata.get("duration"),
            "total_num_frames": int(metadata["total_num_frames"]),
            "frames_indices": list(metadata.get("frames_indices") or []),
            "video_backend": str(metadata.get("video_backend") or "opencv"),
            "height": metadata.get("height"),
            "width": metadata.get("width"),
            "do_sample_frames": False,
        },
    )


def _resolve_video_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise ValueError(f"BridgeProt video packing expects file:// URLs, but got: {url}")
    return Path(unquote(parsed.path))

def _to_video_metadata(metadata: dict[str, object]) -> VideoMetadata:
    return VideoMetadata(
        total_num_frames=int(metadata["total_num_frames"]),
        fps=float(metadata["fps"]) if metadata.get("fps") is not None else None,
        duration=float(metadata["duration"]) if metadata.get("duration") is not None else None,
        width=int(metadata["width"]) if metadata.get("width") is not None else None,
        height=int(metadata["height"]) if metadata.get("height") is not None else None,
        frames_indices=list(metadata.get("frames_indices") or []),
    )
