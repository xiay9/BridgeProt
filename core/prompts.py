from __future__ import annotations

from core.config import BridgeProtProtocolConfig
from data.schema import Dialogue


SYSTEM_PROMPT = (
    "You are a careful MECPE annotation assistant. "
    "You must output only valid JSON that follows the requested schema. "
    "Do not reveal reasoning or thinking process."
)


def _build_output_format(output_mode: str) -> str:
    if output_mode == "minimal":
        return """{
  "records": [
    {
      "emotion_turn": <int>,
      "evidence": [<int>, ...]
    }
  ]
}"""
    return """{
  "records": [
    {
      "emotion_turn": <int>,
      "evidence": [<int>, ...],
      "bridge": <string or null>,
      "explanation": <string>
    }
  ]
}"""


def _build_rules(
    *,
    dialogue: Dialogue,
    protocol: BridgeProtProtocolConfig,
    output_mode: str,
) -> str:
    rules = [
        "1. Each record corresponds to one emotion turn that expresses a non-neutral emotion.",
        '2. "evidence" contains the cause turn indices for that emotion turn.',
        f"3. Turn indices are 1-based. Every emotion turn and cause turn must be between 1 and {dialogue.num_utterances}.",
        "4. A cause turn may be earlier than, equal to, or later than the emotion turn.",
        "5. Keep evidence unique and sorted ascending.",
        f"6. Use at most {protocol.max_records} records in total.",
        f"7. Use at most {protocol.max_evidence_per_record} evidence turns per record.",
    ]
    if output_mode == "minimal":
        rules.extend(
            [
                '8. Do not output "bridge" or "explanation" fields.',
                '9. If there is no valid emotion-cause pair, output {"records": []}.',
                "10. Return JSON only. Do not add markdown, comments, or extra text.",
            ]
        )
    else:
        rules.extend(
            [
                '8. Set "bridge" to null when no implicit bridge is needed.',
                '9. Keep "bridge" concise and keep "explanation" concise.',
                '10. If there is no valid emotion-cause pair, output {"records": []}.',
                "11. Return JSON only. Do not add markdown, comments, or extra text.",
            ]
        )
    return "\n".join(rules)


def build_bridgeprot_user_prompt(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
    audio_lines: list[str] | None = None,
) -> str:
    audio_block = ""
    if audio_lines:
        audio_block = "\n\nAudio Summaries:\n" + "\n".join(audio_lines)

    return f"""You are given one dialogue with numbered turns.

Your task is to extract bridgeable emotion-cause explanations from the dialogue.

Output format:
{_build_output_format(output_mode)}

Rules:
{_build_rules(dialogue=dialogue, protocol=protocol, output_mode=output_mode)}

Dialogue:
{serialized_dialogue}{audio_block}
"""


def build_bridgeprot_user_content(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
    use_video: bool = False,
    audio_lines: list[str] | None = None,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": build_bridgeprot_user_prompt(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
                output_mode=output_mode,
                audio_lines=audio_lines,
            ),
        }
    ]
    if not use_video:
        return content

    seen_video_urls: set[str] = set()
    appended_native_video = False
    for utterance in dialogue.utterances:
        video_url = utterance.metadata.get("video_url")
        if not video_url or video_url in seen_video_urls:
            continue
        content.append({"type": "text", "text": f"[Turn Video] turn={utterance.turn}"})
        content.append({"type": "video_url", "video_url": {"url": video_url}})
        seen_video_urls.add(str(video_url))
        appended_native_video = True

    if appended_native_video:
        return content

    dialogue_video = dialogue.metadata.get("video_url")
    if dialogue_video:
        content.append(
            {
                "type": "text",
                "text": "[Dialogue Video] The following clip covers the wider dialogue context.",
            }
        )
        content.append({"type": "video_url", "video_url": {"url": dialogue_video}})
    return content


def build_bridgeprot_chat_messages(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
    use_video: bool = False,
    audio_lines: list[str] | None = None,
) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_bridgeprot_user_content(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
                output_mode=output_mode,
                use_video=use_video,
                audio_lines=audio_lines,
            ),
        },
    ]


def build_bridge_window_prompt(
    *,
    serialized_context: str,
    emotion_turn: int,
    candidate_turns: list[int],
    seeded_cause_turns: list[int],
    audio_lines: list[str],
    output_field_name: str = "cause_turns",
    ranked_output: bool = False,
) -> str:
    if ranked_output:
        return_line = (
            f'Return JSON only with fields: {{"emotion_supported": <true|false>, '
            f'"{output_field_name}": [<int>, ...]}}.'
        )
        subset_line = (
            f"The output {output_field_name} must be a unique subset of the candidate cause turns "
            "ordered from most likely cause to least likely cause."
        )
        empty_line = (
            f'If the emotion turn is unsupported or no candidate turn is a valid cause, return '
            f'{{"emotion_supported": false, "{output_field_name}": []}}.'
        )
    else:
        return_line = 'Return JSON only with fields: {"emotion_supported": <true|false>, "cause_turns": [<int>, ...]}.'
        subset_line = "The output cause_turns must be a unique sorted subset of the candidate cause turns."
        empty_line = 'If the emotion turn is unsupported or no candidate turn is a valid cause, return {"emotion_supported": false, "cause_turns": []}.'

    sections = [
        "[Dialogue Context]",
        serialized_context,
    ]
    if audio_lines:
        sections.extend(
            [
                "",
                "[Audio Cues]",
                "\n".join(audio_lines),
            ]
        )
    sections.extend(
        [
            "",
            "[Task]",
            f"Emotion turn under review: {emotion_turn}",
            f"Candidate cause turns in the local window: {candidate_turns}",
            f"Stage-1 suggested cause turns in this window: {seeded_cause_turns}",
            "First decide whether the emotion turn should remain in the final MECPE output.",
            "The stage-1 emotion turn itself may be a false positive. If the turn does not clearly express a non-neutral emotion with a valid local cause in this window, mark it unsupported and return no causes.",
            "Review the candidate turns and keep only the turns that should remain as valid causes of the emotion expressed in the emotion turn.",
            "Stage-1 suggestions are only optional references. Keep a suggested cause only if the local evidence supports it. Drop it if it looks unsupported.",
            "Be conservative: if a candidate turn is weak, ambiguous, or only indirectly related, do not select it.",
            "Most valid causes are in the same turn or an earlier turn. A later turn is rare and should be selected only when the local evidence is strong and direct.",
            "In this task, most emotion turns have at most 3 valid local causes.",
            return_line,
            subset_line,
            empty_line,
        ]
    )
    return "\n".join(section for section in sections if section is not None)


def build_bridgeprot_messages(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    output_mode: str = "minimal",
    audio_lines: list[str] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_bridgeprot_user_prompt(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
                output_mode=output_mode,
                audio_lines=audio_lines,
            ),
        },
    ]
