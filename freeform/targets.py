from __future__ import annotations

from core.config import BridgeProtProtocolConfig
from data.targets import build_bridgeprot_target_output
from data.schema import Dialogue


def render_freeform_target_text(
    dialogue: Dialogue,
    *,
    protocol: BridgeProtProtocolConfig,
) -> str:
    output = build_bridgeprot_target_output(dialogue, protocol=protocol)
    if not output.records:
        return "No valid emotion-cause pairs."

    lines: list[str] = []
    for record in output.records:
        if len(record.evidence) == 1:
            lines.append(
                f"Emotion turn {record.emotion_turn}: cause turn {record.evidence[0]}."
            )
        else:
            evidence_text = ", ".join(str(item) for item in record.evidence)
            lines.append(
                f"Emotion turn {record.emotion_turn}: cause turns {evidence_text}."
            )
    return "\n".join(lines)
