from __future__ import annotations

import json

from core.config import BridgeProtProtocolConfig
from core.schema import BridgeDialogueOutput, BridgeRecord
from data.schema import Dialogue


def build_bridgeprot_target_output(
    dialogue: Dialogue,
    *,
    protocol: BridgeProtProtocolConfig,
) -> BridgeDialogueOutput:
    grouped: dict[int, list[int]] = {}
    for emotion_turn, cause_turn in dialogue.emotion_cause_pairs:
        grouped.setdefault(emotion_turn, []).append(cause_turn)

    records: list[BridgeRecord] = []
    for emotion_turn in sorted(grouped):
        canonical_evidence = sorted(
            {
                cause_turn
                for cause_turn in grouped[emotion_turn]
                if 1 <= cause_turn <= dialogue.num_utterances
            }
        )
        for evidence_chunk in _chunk(canonical_evidence, protocol.max_evidence_per_record):
            records.append(
                BridgeRecord(
                    emotion_turn=emotion_turn,
                    evidence=evidence_chunk,
                    bridge=None,
                    explanation=_build_templated_explanation(
                        emotion_turn=emotion_turn,
                        evidence=evidence_chunk,
                    ),
                )
            )

    if len(records) > protocol.max_records:
        records = records[: protocol.max_records]
    return BridgeDialogueOutput(records=records)


def render_bridgeprot_target_json(
    dialogue: Dialogue,
    *,
    protocol: BridgeProtProtocolConfig,
    target_mode: str = "full",
) -> str:
    output = build_bridgeprot_target_output(dialogue, protocol=protocol)
    if target_mode == "minimal":
        payload = {
            "records": [
                {
                    "emotion_turn": record.emotion_turn,
                    "evidence": record.evidence,
                }
                for record in output.records
            ]
        }
    else:
        payload = {
            "records": [
                {
                    "emotion_turn": record.emotion_turn,
                    "evidence": record.evidence,
                    "bridge": record.bridge,
                    "explanation": record.explanation,
                }
                for record in output.records
            ]
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _chunk(values: list[int], chunk_size: int) -> list[list[int]]:
    if not values:
        return []
    if chunk_size <= 0:
        return [values]
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _build_templated_explanation(*, emotion_turn: int, evidence: list[int]) -> str:
    if len(evidence) == 1:
        return f"Turn {evidence[0]} supports the emotion expressed at turn {emotion_turn}."
    evidence_text = ", ".join(str(item) for item in evidence)
    return f"Turns {evidence_text} support the emotion expressed at turn {emotion_turn}."
