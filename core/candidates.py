from __future__ import annotations

import json
from dataclasses import dataclass

from core.config import BridgeProtMethodConfig, BridgeProtProtocolConfig


@dataclass(slots=True)
class EmotionWindowCandidate:
    emotion_turn: int
    candidate_turns: list[int]
    context_turns: list[int]
    seeded_cause_turns: list[int]


def build_emotion_windows(
    *,
    num_turns: int,
    seed_pairs: set[tuple[int, int]],
    method_config: BridgeProtMethodConfig,
    protocol_config: BridgeProtProtocolConfig,
) -> list[EmotionWindowCandidate]:
    emotion_turns = sorted({emotion_turn for emotion_turn, _ in seed_pairs})
    if not emotion_turns and method_config.fallback_to_window_scan:
        emotion_turns = list(range(1, num_turns + 1))

    if method_config.max_emotion_windows is not None:
        emotion_turns = emotion_turns[: method_config.max_emotion_windows]

    windows: list[EmotionWindowCandidate] = []
    for emotion_turn in emotion_turns:
        start = max(1, emotion_turn - method_config.candidate_window)
        end = min(num_turns, emotion_turn + method_config.candidate_window)
        candidate_turns = [
            cause_turn
            for cause_turn in range(start, end + 1)
            if not (protocol_config.enforce_temporal_precedence and cause_turn > emotion_turn)
        ]
        seeded_cause_turns = sorted(
            {
                cause_turn
                for seed_emotion_turn, cause_turn in seed_pairs
                if seed_emotion_turn == emotion_turn and cause_turn in set(candidate_turns)
            }
        )
        windows.append(
            EmotionWindowCandidate(
                emotion_turn=emotion_turn,
                candidate_turns=candidate_turns,
                context_turns=build_emotion_window_context_turns(
                    num_turns=num_turns,
                    emotion_turn=emotion_turn,
                    candidate_turns=candidate_turns,
                    max_context_turns=method_config.max_context_turns,
                ),
                seeded_cause_turns=seeded_cause_turns,
            )
        )
    return windows


def build_emotion_window_context_turns(
    *,
    num_turns: int,
    emotion_turn: int,
    candidate_turns: list[int],
    max_context_turns: int,
) -> list[int]:
    if num_turns <= 0:
        return []

    if candidate_turns:
        start = min(candidate_turns)
        end = max(candidate_turns)
    else:
        start = emotion_turn
        end = emotion_turn
    target = min(num_turns, max(max_context_turns, end - start + 1))
    while (end - start + 1) < target:
        if start > 1:
            start -= 1
        if (end - start + 1) >= target:
            break
        if end < num_turns:
            end += 1
        if start == 1 and end == num_turns:
            break
    return list(range(start, end + 1))


def render_bridgeprot_prediction_json(
    *,
    accepted_pairs: set[tuple[int, int]],
    protocol: BridgeProtProtocolConfig,
    output_mode: str,
) -> str:
    grouped: dict[int, list[int]] = {}
    for emotion_turn, cause_turn in sorted(accepted_pairs):
        grouped.setdefault(int(emotion_turn), []).append(int(cause_turn))

    records: list[dict[str, object]] = []
    for emotion_turn in sorted(grouped):
        canonical_evidence = sorted(set(grouped[emotion_turn]))
        for start in range(0, len(canonical_evidence), protocol.max_evidence_per_record):
            evidence_chunk = canonical_evidence[start : start + protocol.max_evidence_per_record]
            record = {
                "emotion_turn": emotion_turn,
                "evidence": evidence_chunk,
            }
            if output_mode == "full":
                record["bridge"] = None
                if len(evidence_chunk) == 1:
                    record["explanation"] = (
                        f"Turn {evidence_chunk[0]} supports the emotion expressed at turn {emotion_turn}."
                    )
                else:
                    evidence_text = ", ".join(str(item) for item in evidence_chunk)
                    record["explanation"] = (
                        f"Turns {evidence_text} support the emotion expressed at turn {emotion_turn}."
                    )
            records.append(record)

    if len(records) > protocol.max_records:
        records = records[: protocol.max_records]
    return json.dumps({"records": records}, ensure_ascii=False)
