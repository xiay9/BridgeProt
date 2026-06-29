from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(
    path: str | Path,
    *,
    num_frames: int,
    max_edge: int,
) -> tuple[np.ndarray, dict[str, object]]:
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1, 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else None
    duration = (float(total_frames) / fps) if fps and fps > 0 else None
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None

    sampled_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int).tolist()
    sampled_frames: list[np.ndarray] = []
    resolved_indices: list[int] = []
    for frame_index in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_frame(frame, max_edge=max_edge)
        sampled_frames.append(frame)
        resolved_indices.append(int(frame_index))
    cap.release()

    if not sampled_frames:
        raise RuntimeError(f"Failed to decode any frames from video: {video_path}")

    if len(sampled_frames) < num_frames:
        sampled_frames.extend([sampled_frames[-1]] * (num_frames - len(sampled_frames)))
        resolved_indices.extend([resolved_indices[-1]] * (num_frames - len(resolved_indices)))

    frames = np.stack(sampled_frames, axis=0).astype(np.uint8, copy=False)
    metadata = {
        "total_num_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "frames_indices": resolved_indices,
        "video_backend": "opencv",
        "height": original_height if original_height is not None else int(frames.shape[1]),
        "width": original_width if original_width is not None else int(frames.shape[2]),
    }
    return frames, metadata


def _resize_frame(frame: np.ndarray, *, max_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= 0:
        return frame
    scale = min(float(max_edge) / float(longest), 1.0)
    if scale == 1.0:
        return frame
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
