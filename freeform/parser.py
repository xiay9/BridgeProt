from __future__ import annotations

from dataclasses import dataclass, field
import re

from core.config import BridgeProtProtocolConfig
from core.schema import BridgeRecord


_NO_PAIR_PATTERNS = (
    re.compile(r"\bno valid emotion[- ]cause pairs\b", flags=re.IGNORECASE),
    re.compile(r"\bno emotion[- ]cause pairs\b", flags=re.IGNORECASE),
)

_RECORD_PATTERN = re.compile(
    r"emotion\s*turn\s*(?P<emotion>\d+)\s*:\s*cause\s*turns?\s*(?P<causes>[0-9,\sand]+)",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class FreeformParseResult:
    parsed: bool
    records: list[BridgeRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    invalid_record_indices: list[int] = field(default_factory=list)
    matched_record_count: int = 0


def extract_freeform_records(
    raw_text: str,
    *,
    num_turns: int,
    protocol: BridgeProtProtocolConfig,
) -> FreeformParseResult:
    stripped = raw_text.strip()
    if not stripped:
        return FreeformParseResult(parsed=False, errors=["empty_response"])

    for pattern in _NO_PAIR_PATTERNS:
        if pattern.search(stripped):
            return FreeformParseResult(parsed=True)

    errors: list[str] = []
    records: list[BridgeRecord] = []
    invalid_record_indices: list[int] = []
    matched_record_count = 0
    for index, match in enumerate(_RECORD_PATTERN.finditer(stripped)):
        matched_record_count += 1
        emotion_turn = int(match.group("emotion"))
        if not (1 <= emotion_turn <= num_turns):
            errors.append(f"invalid_emotion_turn:{emotion_turn}")
            invalid_record_indices.append(index)
            continue
        cause_turns = sorted(
            {
                int(item)
                for item in re.findall(r"\d+", match.group("causes"))
                if 1 <= int(item) <= num_turns
            }
        )
        if not cause_turns:
            errors.append(f"missing_cause_turns:{emotion_turn}")
            invalid_record_indices.append(index)
            continue
        records.append(
            BridgeRecord(
                emotion_turn=emotion_turn,
                evidence=cause_turns[: protocol.max_evidence_per_record],
                bridge=None,
                explanation="",
            )
        )

    if len(records) > protocol.max_records:
        records = records[: protocol.max_records]
        errors.append("too_many_records")

    return FreeformParseResult(
        parsed=bool(records),
        records=records,
        errors=errors,
        invalid_record_indices=invalid_record_indices,
        matched_record_count=matched_record_count,
    )
