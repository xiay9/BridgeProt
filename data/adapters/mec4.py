from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from data.adapters.base import DatasetAdapter
from data.schema import Dialogue, LoadedSplit, Utterance


PAIR_PATTERN = re.compile(r"[（(](\d+)[，,](\d+)[）)]")
TURN_PATTERN = re.compile(r"(\d+)\.(.*?)(?=(?:\s+\d+\.)|$)")
TURN_EMOTION_PATTERN = re.compile(
    r"(?:^|[|;,，；\s])(?:u|utt|utterance|turn)?\s*(\d+)\s*[:：=,，]\s*([A-Za-z_ -]+)",
    flags=re.IGNORECASE,
)
EMOTION_TURN_PATTERN = re.compile(
    r"(?:^|[|;,，；\s])([A-Za-z_ -]+)\s*[:：=,，]\s*(?:u|utt|utterance|turn)?\s*(\d+)",
    flags=re.IGNORECASE,
)

EMOTION_KEY_CANDIDATES = (
    "emotion",
    "emotions",
    "emotion_label",
    "emotion_labels",
    "emotion_category",
    "emotion_categories",
)
UTTERANCE_KEY_CANDIDATES = ("utterances", "dialog", "dialogue", "conversation")
TEXT_KEY_CANDIDATES = ("text", "utterance", "sentence", "content")
SPEAKER_KEY_CANDIDATES = ("speaker", "speaker_name", "role")


def _canonical_emotion(label: Any) -> str | None:
    text = str(label or "").strip()
    if not text:
        return None
    key = text.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "neutral": "neutral",
        "neu": "neutral",
        "anger": "anger",
        "angry": "anger",
        "ang": "anger",
        "surprise": "surprise",
        "surprised": "surprise",
        "sur": "surprise",
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "sad": "sad",
        "sadness": "sad",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "emotion": "emotion",
    }
    return mapping.get(key, key)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


class MEC4Adapter(DatasetAdapter):
    dataset_name = "MEC4"
    available_splits = ("train", "valid", "test")
    split_aliases = {"valid": ("valid", "dev")}

    def split_file(self, split: str) -> Path:
        split = self.canonical_split(split)
        file_map = {"train": "train.json", "valid": "valid.json", "test": "test.json"}
        return self.root / "MEC4_text" / file_map[split]

    def parse_pairs(self, raw_pairs: Any) -> list[tuple[int, int]]:
        if isinstance(raw_pairs, list):
            pairs: list[tuple[int, int]] = []
            for item in raw_pairs:
                if isinstance(item, dict):
                    target = item.get("emotion_turn", item.get("target_turn", item.get("turn")))
                    cause = item.get("cause_turn", item.get("candidate_turn", item.get("cause")))
                    if target is not None and cause is not None:
                        pairs.append((int(target), int(cause)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((int(item[0]), int(item[1])))
            return pairs
        return [(int(src), int(dst)) for src, dst in PAIR_PATTERN.findall(str(raw_pairs or ""))]

    def parse_utterances(
        self,
        raw_text: str,
        emotions_by_turn: dict[int, str] | None = None,
    ) -> list[Utterance]:
        emotions_by_turn = emotions_by_turn or {}
        utterances: list[Utterance] = []
        for match in TURN_PATTERN.finditer(raw_text.strip()):
            turn = int(match.group(1))
            content = match.group(2).strip()
            if "：" in content:
                speaker, text = content.split("：", 1)
            elif ":" in content:
                speaker, text = content.split(":", 1)
            else:
                speaker, text = "", content
            utterances.append(
                Utterance(
                    turn=turn,
                    speaker=speaker.strip(),
                    text=text.strip(),
                    emotion=emotions_by_turn.get(turn),
                )
            )
        return utterances

    def parse_structured_utterances(
        self,
        rows: list[Any],
        emotions_by_turn: dict[int, str] | None = None,
    ) -> list[Utterance]:
        emotions_by_turn = emotions_by_turn or {}
        utterances: list[Utterance] = []
        for index, item in enumerate(rows, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported MEC4 utterance row type: {type(item)}")
            turn = int(item.get("turn", item.get("id", index)))
            emotion = _canonical_emotion(_first_present(item, EMOTION_KEY_CANDIDATES))
            utterances.append(
                Utterance(
                    turn=turn,
                    speaker=str(_first_present(item, SPEAKER_KEY_CANDIDATES) or "").strip(),
                    text=str(_first_present(item, TEXT_KEY_CANDIDATES) or "").strip(),
                    emotion=emotion or emotions_by_turn.get(turn),
                    timecode=item.get("timecode"),
                    utterance_name=item.get("utterance_name"),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "turn",
                            "id",
                            *TEXT_KEY_CANDIDATES,
                            *SPEAKER_KEY_CANDIDATES,
                            *EMOTION_KEY_CANDIDATES,
                            "timecode",
                            "utterance_name",
                        }
                    },
                )
            )
        return utterances

    def parse_emotions(self, raw_emotions: Any, dia_len: int | None = None) -> dict[int, str]:
        if raw_emotions is None:
            return {}

        if isinstance(raw_emotions, dict):
            nested = _first_present(raw_emotions, EMOTION_KEY_CANDIDATES)
            if nested is not None:
                return self.parse_emotions(nested, dia_len=dia_len)
            out: dict[int, str] = {}
            for key, value in raw_emotions.items():
                try:
                    turn = int(str(key).strip().lstrip("Uu"))
                except ValueError:
                    continue
                emotion = _canonical_emotion(value)
                if emotion:
                    out[turn] = emotion
            return out

        if isinstance(raw_emotions, list):
            if all(isinstance(item, str) for item in raw_emotions):
                if dia_len is None or len(raw_emotions) == dia_len:
                    return {
                        turn: emotion
                        for turn, label in enumerate(raw_emotions, start=1)
                        if (emotion := _canonical_emotion(label))
                    }
                out: dict[int, str] = {}
                for item in raw_emotions:
                    out.update(self.parse_emotions(item))
                return out
            out: dict[int, str] = {}
            for index, item in enumerate(raw_emotions, start=1):
                if isinstance(item, dict):
                    turn = int(item.get("turn", item.get("id", index)))
                    emotion = _canonical_emotion(_first_present(item, EMOTION_KEY_CANDIDATES))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    turn = int(item[0])
                    emotion = _canonical_emotion(item[1])
                else:
                    continue
                if emotion:
                    out[turn] = emotion
            return out

        if isinstance(raw_emotions, str):
            text = raw_emotions.strip()
            if not text:
                return {}
            if text[0] in "[{":
                try:
                    return self.parse_emotions(json.loads(text), dia_len=dia_len)
                except json.JSONDecodeError:
                    pass

            out: dict[int, str] = {}
            for pattern in (TURN_EMOTION_PATTERN, EMOTION_TURN_PATTERN):
                for match in pattern.finditer(text):
                    first, second = match.group(1), match.group(2)
                    if pattern is TURN_EMOTION_PATTERN:
                        turn_text, label = first, second
                    else:
                        label, turn_text = first, second
                    emotion = _canonical_emotion(label)
                    if emotion:
                        out[int(turn_text)] = emotion
            if out:
                return out

            labels = [
                _canonical_emotion(part)
                for part in re.split(r"\s*[|;；]\s*", text)
                if part.strip()
            ]
            labels = [label for label in labels if label]
            if labels and (dia_len is None or len(labels) == dia_len):
                return {turn: label for turn, label in enumerate(labels, start=1)}

        return {}

    def row_utterances(self, row: dict[str, Any]) -> tuple[list[Utterance], dict[str, Any]]:
        dia_len = int(row.get("dia_len", 0) or 0)
        emotions_by_turn = self.parse_emotions(
            _first_present(row, EMOTION_KEY_CANDIDATES),
            dia_len=dia_len or None,
        )
        utterance_rows = _first_present(row, UTTERANCE_KEY_CANDIDATES)
        if isinstance(utterance_rows, list):
            utterances = self.parse_structured_utterances(utterance_rows, emotions_by_turn)
        else:
            utterances = self.parse_utterances(row["text"], emotions_by_turn)

        metadata = {
            "mec4_emotion_labels_present": any(utterance.emotion for utterance in utterances),
            "mec4_emotion_source": "row" if emotions_by_turn else ("utterances" if isinstance(utterance_rows, list) else None),
        }
        return utterances, metadata

    def apply_pair_binary_emotions(
        self,
        utterances: list[Utterance],
        pairs: list[tuple[int, int]],
    ) -> bool:
        if any(utterance.emotion for utterance in utterances):
            return False
        emotion_turns = {int(emotion_turn) for emotion_turn, _cause_turn in pairs}
        for utterance in utterances:
            utterance.emotion = "emotion" if int(utterance.turn) in emotion_turns else "neutral"
        return True

    def load_split(self, split: str) -> LoadedSplit:
        split = self.canonical_split(split)
        path = self.split_file(split)
        rows = json.loads(path.read_text(encoding="utf-8"))
        video_index = self._build_video_index()

        dialogues = []
        for row in rows:
            pairs = self.parse_pairs(row.get("pairs", row.get("emotion_cause_pairs", "")))
            utterances, label_metadata = self.row_utterances(row)
            if self.apply_pair_binary_emotions(utterances, pairs):
                label_metadata = {
                    **label_metadata,
                    "mec4_emotion_labels_present": any(
                        utterance.emotion == "emotion" for utterance in utterances
                    ),
                    "mec4_emotion_source": "pairs_binary",
                }
            dialogue_id = str(row.get("dia_id", row.get("dialogue_id", "")))
            if not dialogue_id:
                raise ValueError(f"MEC4 row is missing dia_id/dialogue_id in {path}")
            raw_text = row.get("text")
            if raw_text is None:
                raw_text = "\n".join(f"{utterance.turn}.{utterance.speaker}：{utterance.text}" for utterance in utterances)
            dialogues.append(
                Dialogue(
                    dataset=self.dataset_name,
                    split=split,
                    dialogue_id=dialogue_id,
                    utterances=utterances,
                    emotion_cause_pairs=pairs,
                    raw_text=str(raw_text),
                    metadata={
                        "declared_dia_len": row.get("dia_len", len(utterances)),
                        "source_file": str(path),
                        **label_metadata,
                        **(
                            {"video_path": str(video_index[dialogue_id].resolve())}
                            if dialogue_id in video_index
                            else {}
                        ),
                    },
                )
            )

        metadata = {"source_file": str(path)}
        subtitles_path = self.root / "subtitles.json"
        if subtitles_path.exists():
            metadata["subtitles_file"] = str(subtitles_path)

        return LoadedSplit(
            dataset=self.dataset_name,
            split=split,
            root=self.root,
            dialogues=dialogues,
            feature_paths=self.resolve_feature_paths(split),
            metadata=metadata,
        )

    def _build_video_index(self) -> dict[str, Path]:
        root = self.root / "dialogue_videos"
        if not root.exists():
            return {}
        return {path.stem: path for path in root.rglob("*.mp4")}
