from __future__ import annotations

from dataclasses import dataclass, field

from core.config import BridgeProtProtocolConfig


@dataclass(slots=True)
class BridgeRecord:
    emotion_turn: int
    evidence: list[int]
    bridge: str | None
    explanation: str


@dataclass(slots=True)
class BridgeDialogueOutput:
    records: list[BridgeRecord] = field(default_factory=list)


def build_bridgeprot_json_schema(
    protocol: BridgeProtProtocolConfig,
    *,
    output_mode: str = "minimal",
) -> dict:
    if output_mode not in {"minimal", "full"}:
        raise ValueError("BridgeProt output_mode must be either 'minimal' or 'full'.")

    bridge_schema: dict = {
        "type": ["string", "null"],
        "maxLength": protocol.max_bridge_chars,
    }
    explanation_schema: dict = {"type": "string"}
    if protocol.require_explanation:
        explanation_schema["minLength"] = 1
    explanation_schema["maxLength"] = protocol.max_explanation_chars

    record_properties: dict = {
        "emotion_turn": {
            "type": "integer",
            "minimum": 1,
        },
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": protocol.max_evidence_per_record,
            "items": {
                "type": "integer",
                "minimum": 1,
            },
        },
    }
    required_fields = [
        "emotion_turn",
        "evidence",
    ]
    if output_mode == "full":
        record_properties["bridge"] = bridge_schema
        record_properties["explanation"] = explanation_schema
        required_fields.extend(["bridge", "explanation"])

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "records": {
                "type": "array",
                "maxItems": protocol.max_records,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": record_properties,
                    "required": required_fields,
                },
            },
        },
        "required": ["records"],
    }


def build_emotion_window_selection_json_schema(
    *,
    max_items: int,
    field_name: str = "cause_turns",
) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "emotion_supported": {
                "type": "boolean",
            },
            field_name: {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
        },
        "required": ["emotion_supported", field_name],
    }
