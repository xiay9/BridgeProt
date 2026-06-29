from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.config import BridgeProtProtocolConfig
from core.schema import BridgeDialogueOutput, BridgeRecord


@dataclass(slots=True)
class BridgeValidationResult:
    raw_text: str
    parsed: bool
    parse_strategy: str
    strict_valid: bool
    strict_output: BridgeDialogueOutput | None
    salvaged_output: BridgeDialogueOutput
    errors: list[str] = field(default_factory=list)
    invalid_record_indices: list[int] = field(default_factory=list)


def _extract_json_substring(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            content_lines = lines[1:]
            if content_lines and content_lines[-1].strip() == "```":
                content_lines = content_lines[:-1]
            return "\n".join(content_lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _parse_json_payload(text: str) -> tuple[Any | None, str, list[str]]:
    errors: list[str] = []
    stripped = text.strip()
    if not stripped:
        return None, "empty", ["empty_response"]

    try:
        return json.loads(stripped), "strict", errors
    except json.JSONDecodeError as exc:
        errors.append(f"strict_json_decode_error:{exc.msg}")

    candidate = _extract_json_substring(text)
    if candidate is None:
        errors.append("no_json_object_found")
        return None, "none", errors

    try:
        return json.loads(candidate), "salvaged", errors
    except json.JSONDecodeError as exc:
        errors.append(f"salvaged_json_decode_error:{exc.msg}")
        return None, "none", errors


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def validate_bridge_output(
    raw_text: str,
    *,
    num_turns: int,
    protocol: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
) -> BridgeValidationResult:
    if output_mode not in {"minimal", "full"}:
        raise ValueError("BridgeProt output_mode must be either 'minimal' or 'full'.")
    payload, parse_strategy, parse_errors = _parse_json_payload(raw_text)
    if payload is None:
        return BridgeValidationResult(
            raw_text=raw_text,
            parsed=False,
            parse_strategy=parse_strategy,
            strict_valid=False,
            strict_output=None,
            salvaged_output=BridgeDialogueOutput(records=[]),
            errors=parse_errors,
            invalid_record_indices=[],
        )

    errors = list(parse_errors)
    if not isinstance(payload, dict):
        errors.append("top_level_not_object")
        return BridgeValidationResult(
            raw_text=raw_text,
            parsed=False,
            parse_strategy=parse_strategy,
            strict_valid=False,
            strict_output=None,
            salvaged_output=BridgeDialogueOutput(records=[]),
            errors=errors,
            invalid_record_indices=[],
        )

    records_payload = payload.get("records")
    if not isinstance(records_payload, list):
        errors.append("records_not_list")
        return BridgeValidationResult(
            raw_text=raw_text,
            parsed=False,
            parse_strategy=parse_strategy,
            strict_valid=False,
            strict_output=None,
            salvaged_output=BridgeDialogueOutput(records=[]),
            errors=errors,
            invalid_record_indices=[],
        )

    if len(records_payload) > protocol.max_records:
        errors.append("too_many_records")

    strict_records: list[BridgeRecord] = []
    salvaged_records: list[BridgeRecord] = []
    invalid_record_indices: list[int] = []

    for index, item in enumerate(records_payload):
        if not isinstance(item, dict):
            errors.append(f"record_{index}_not_object")
            invalid_record_indices.append(index)
            continue

        emotion_turn = _coerce_int(item.get("emotion_turn"))
        evidence_payload = item.get("evidence")
        bridge_value = item.get("bridge")
        explanation_value = item.get("explanation")

        current_errors: list[str] = []
        if emotion_turn is None or not (1 <= emotion_turn <= num_turns):
            current_errors.append("invalid_emotion_turn")

        if not isinstance(evidence_payload, list):
            current_errors.append("evidence_not_list")
            evidence_values: list[int] = []
        else:
            evidence_values = []
            for value in evidence_payload:
                int_value = _coerce_int(value)
                if int_value is None:
                    current_errors.append("non_integer_evidence")
                    continue
                evidence_values.append(int_value)

        if output_mode == "full":
            if bridge_value is not None and not isinstance(bridge_value, str):
                current_errors.append("bridge_not_string_or_null")

            if explanation_value is None:
                if protocol.require_explanation:
                    current_errors.append("missing_explanation")
                explanation_text = ""
            elif not isinstance(explanation_value, str):
                current_errors.append("explanation_not_string")
                explanation_text = ""
            else:
                explanation_text = explanation_value
        else:
            bridge_value = None
            explanation_text = ""

        canonical_evidence = sorted(
            {
                cause_turn
                for cause_turn in evidence_values
                if 1 <= cause_turn <= num_turns
                and (
                    not protocol.enforce_temporal_precedence
                    or (emotion_turn is not None and cause_turn <= emotion_turn)
                )
            }
        )

        if emotion_turn is not None and canonical_evidence:
            salvaged_records.append(
                BridgeRecord(
                    emotion_turn=emotion_turn,
                    evidence=canonical_evidence[: protocol.max_evidence_per_record],
                    bridge=bridge_value if isinstance(bridge_value, str) else None,
                    explanation=explanation_text,
                )
            )

        if len(evidence_values) > protocol.max_evidence_per_record:
            current_errors.append("too_many_evidence_items")
        if output_mode == "full" and len(explanation_text) > protocol.max_explanation_chars:
            current_errors.append("explanation_too_long")
        if output_mode == "full" and isinstance(bridge_value, str) and len(bridge_value) > protocol.max_bridge_chars:
            current_errors.append("bridge_too_long")
        if emotion_turn is not None and not canonical_evidence:
            current_errors.append("empty_canonical_evidence")

        if current_errors:
            errors.extend(f"record_{index}_{err}" for err in current_errors)
            invalid_record_indices.append(index)
            continue

        strict_records.append(
            BridgeRecord(
                emotion_turn=emotion_turn,
                evidence=canonical_evidence[: protocol.max_evidence_per_record],
                bridge=bridge_value if isinstance(bridge_value, str) else None,
                explanation=explanation_text,
            )
        )

    strict_valid = len(errors) == len(parse_errors)
    strict_output = BridgeDialogueOutput(records=strict_records) if strict_valid else None
    return BridgeValidationResult(
        raw_text=raw_text,
        parsed=True,
        parse_strategy=parse_strategy,
        strict_valid=strict_valid,
        strict_output=strict_output,
        salvaged_output=BridgeDialogueOutput(records=salvaged_records),
        errors=errors,
        invalid_record_indices=invalid_record_indices,
    )
