from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path

import cv2

from data.schema import LoadedSplit


def materialize_video_paths(loaded_split: LoadedSplit) -> LoadedSplit:
    clip_cache_root = loaded_split.root / "cache" / "utterance_video_clips"

    for dialogue in loaded_split.dialogues:
        source_video_path = _coerce_path(dialogue.metadata.get("source_video_path"))
        dialogue_video_path = _coerce_path(dialogue.metadata.get("video_path"))
        if dialogue_video_path is not None:
            dialogue.metadata["video_path"] = str(dialogue_video_path.resolve())

        if source_video_path is None:
            for utterance in dialogue.utterances:
                utterance_video_path = _coerce_path(utterance.metadata.get("video_path"))
                if utterance_video_path is not None:
                    utterance.metadata["video_path"] = str(utterance_video_path.resolve())
            continue

        source_video_path = source_video_path.resolve()
        dialogue.metadata["source_video_path"] = str(source_video_path)

        for utterance in dialogue.utterances:
            utterance_video_path = _coerce_path(utterance.metadata.get("video_path"))
            if utterance_video_path is None:
                start_sec = _coerce_float(utterance.metadata.get("start_sec"))
                end_sec = _coerce_float(utterance.metadata.get("end_sec"))
                if start_sec is not None and end_sec is not None:
                    clip_id = utterance.utterance_name or f"{dialogue.dialogue_id}_{utterance.turn}"
                    utterance_video_path = prepare_utterance_video_clip(
                        source_video_path,
                        cache_root=clip_cache_root / dialogue.dialogue_id,
                        clip_id=clip_id,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
            if utterance_video_path is not None:
                utterance.metadata["video_path"] = str(Path(utterance_video_path).resolve())

    return loaded_split


def prepare_utterance_video_clip(
    source_video_path: str | Path,
    *,
    cache_root: str | Path,
    clip_id: str,
    start_sec: float,
    end_sec: float,
    min_duration_sec: float = 0.5,
) -> Path:
    source = Path(source_video_path).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return _prepare_utterance_video_clip_cached(
        str(source),
        str(cache_root),
        _sanitize_filename(clip_id),
        float(start_sec),
        float(end_sec),
        float(min_duration_sec),
    )


@lru_cache(maxsize=65536)
def _prepare_utterance_video_clip_cached(
    source_path: str,
    cache_root: str,
    clip_id: str,
    start_sec: float,
    end_sec: float,
    min_duration_sec: float,
) -> Path:
    source = Path(source_path)
    cache_dir = Path(cache_root)

    start_sec = max(0.0, start_sec)
    end_sec = max(end_sec, start_sec + min_duration_sec)
    start_ms = int(round(start_sec * 1000.0))
    end_ms = int(round(end_sec * 1000.0))

    clip_path = cache_dir / f"{clip_id}__{start_ms:08d}_{end_ms:08d}.mp4"
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return clip_path

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video while preparing clip: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1)
    height = max(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1)
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_sec * fps)))

    tmp_path = cache_dir / f".{clip_path.stem}.tmp.{os.getpid()}.mp4"
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create clip writer under {cache_dir}")

    wrote_any_frame = False
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        while frame_index < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            wrote_any_frame = True
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if not wrote_any_frame:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to decode any frames while preparing clip {clip_id} from {source}"
        )

    tmp_path.replace(clip_path)
    return clip_path


def prepare_video_summary_clip(
    source_video_path: str | Path,
    *,
    cache_root: str | Path,
    clip_id: str,
    max_duration_sec: float = 50.0,
    segment_duration_sec: float = 15.0,
) -> Path:
    source = Path(source_video_path).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    return _prepare_video_summary_clip_cached(
        str(source),
        str(cache_root),
        _sanitize_filename(clip_id),
        float(max_duration_sec),
        float(segment_duration_sec),
    )


@lru_cache(maxsize=65536)
def _prepare_video_summary_clip_cached(
    source_path: str,
    cache_root: str,
    clip_id: str,
    max_duration_sec: float,
    segment_duration_sec: float,
) -> Path:
    source = Path(source_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video while preparing summary clip: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    if duration_sec <= max_duration_sec:
        cap.release()
        return source

    width = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1)
    height = max(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1)
    clip_path = Path(cache_root) / f"{clip_id}__summary_{int(round(duration_sec * 1000.0)):08d}.mp4"
    if clip_path.exists() and clip_path.stat().st_size > 0:
        cap.release()
        return clip_path

    segment_duration_sec = min(segment_duration_sec, duration_sec)
    middle_start_sec = max(0.0, min(duration_sec - segment_duration_sec, duration_sec / 2.0 - segment_duration_sec / 2.0))
    last_start_sec = max(0.0, duration_sec - segment_duration_sec)
    segments = (
        (0.0, segment_duration_sec),
        (middle_start_sec, middle_start_sec + segment_duration_sec),
        (last_start_sec, last_start_sec + segment_duration_sec),
    )

    tmp_path = clip_path.parent / f".{clip_path.stem}.tmp.{os.getpid()}.mp4"
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to create summary clip writer under {cache_root}")

    wrote_any_frame = False
    try:
        for start_sec, end_sec in segments:
            start_frame = max(0, int(math.floor(start_sec * fps)))
            end_frame = max(start_frame + 1, int(math.ceil(end_sec * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
            while frame_index < end_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                wrote_any_frame = True
                frame_index += 1
    finally:
        cap.release()
        writer.release()

    if not wrote_any_frame:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to decode any frames while preparing summary clip {clip_id} from {source}"
        )

    tmp_path.replace(clip_path)
    return clip_path


def _coerce_path(value: object) -> Path | None:
    if value is None:
        return None
    try:
        path = Path(str(value)).expanduser()
    except (TypeError, ValueError):
        return None
    return path if path.exists() else None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_filename(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return sanitized.strip("_") or "clip"
