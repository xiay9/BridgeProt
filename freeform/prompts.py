from __future__ import annotations

from core.config import BridgeProtProtocolConfig
from data.schema import Dialogue


FREEFORM_SYSTEM_PROMPT = (
    "You are a careful MECPE annotation assistant. "
    "Return plain text only. "
    "Do not reveal your reasoning process."
)


def _build_freeform_rules(
    *,
    dialogue: Dialogue,
    protocol: BridgeProtProtocolConfig,
) -> str:
    return "\n".join(
        [
            "1. Identify non-neutral emotion turns and the dialogue turns that support them as causes.",
            f"2. Turn indices are 1-based and must be between 1 and {dialogue.num_utterances}.",
            "3. A cause turn may be earlier than, equal to, or later than the emotion turn.",
            "4. Keep cause turns unique and sorted ascending.",
            f"5. Use at most {protocol.max_records} records total.",
            f"6. Use at most {protocol.max_evidence_per_record} cause turns per record.",
            '7. Output one line per record using exactly one of these forms:',
            '   Emotion turn X: cause turn Y.',
            '   Emotion turn X: cause turns Y, Z.',
            '8. Do not output JSON, markdown, bullets, numbered lists, or commentary.',
            '9. If there is no valid pair, output exactly: No valid emotion-cause pairs.',
        ]
    )


def build_freeform_user_prompt(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    audio_lines: list[str] | None = None,
) -> str:
    audio_block = ""
    if audio_lines:
        audio_block = "\n\nAudio Summaries:\n" + "\n".join(audio_lines)

    return f"""You are given one dialogue with numbered turns.

Your task is to identify emotion-cause pairs in plain text.

Rules:
{_build_freeform_rules(dialogue=dialogue, protocol=protocol)}

Dialogue:
{serialized_dialogue}{audio_block}
"""


def build_freeform_user_content(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    use_video: bool = False,
    audio_lines: list[str] | None = None,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": build_freeform_user_prompt(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
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


def build_freeform_chat_messages(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    use_video: bool = False,
    audio_lines: list[str] | None = None,
) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": FREEFORM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_freeform_user_content(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
                use_video=use_video,
                audio_lines=audio_lines,
            ),
        },
    ]


def build_freeform_messages(
    dialogue: Dialogue,
    serialized_dialogue: str,
    *,
    protocol: BridgeProtProtocolConfig,
    audio_lines: list[str] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FREEFORM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_freeform_user_prompt(
                dialogue,
                serialized_dialogue,
                protocol=protocol,
                audio_lines=audio_lines,
            ),
        },
    ]
