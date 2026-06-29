from __future__ import annotations

from dataclasses import dataclass

from core.config import BridgeProtDataConfig, BridgeProtProtocolConfig
from core.modalities import load_bridgeprot_multimodal_split
from core.prompts import build_bridgeprot_user_content
from data.targets import render_bridgeprot_target_json
from core.serializer import serialize_dialogue


@dataclass(slots=True)
class BridgeProtDemoExample:
    dataset: str
    split: str
    dialogue_id: str
    serialized_dialogue: str
    target_json: str
    user_content: list[dict[str, object]]
    num_turns: int
    num_speakers: int


def build_demo_bank(
    *,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    source_split: str,
    max_examples: int | None = None,
    output_mode: str = "minimal",
) -> list[BridgeProtDemoExample]:
    loaded_split = load_bridgeprot_multimodal_split(data_config, split=source_split)
    dialogues = loaded_split.dialogues
    if max_examples is not None:
        dialogues = dialogues[:max_examples]

    bank: list[BridgeProtDemoExample] = []
    for dialogue in dialogues:
        serialized = serialize_dialogue(
            dialogue,
            include_turn_id=data_config.include_turn_id,
            include_speaker=data_config.include_speaker,
        )
        audio_lines = (
            _collect_dialogue_audio_summary_lines(dialogue)
            if data_config.audio_summary_enabled
            else []
        )
        bank.append(
            BridgeProtDemoExample(
                dataset=dialogue.dataset,
                split=dialogue.split,
                dialogue_id=dialogue.dialogue_id,
                serialized_dialogue=serialized,
                target_json=render_bridgeprot_target_json(
                    dialogue,
                    protocol=protocol_config,
                    target_mode=output_mode,
                ),
                user_content=build_bridgeprot_user_content(
                    dialogue,
                    serialized,
                    protocol=protocol_config,
                    output_mode=output_mode,
                    use_video=data_config.use_video,
                    audio_lines=audio_lines,
                ),
                num_turns=dialogue.num_utterances,
                num_speakers=len({utterance.speaker for utterance in dialogue.utterances}),
            )
        )
    return bank


def _collect_dialogue_audio_summary_lines(dialogue) -> list[str]:
    lines: list[str] = []
    for utterance in dialogue.utterances:
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            lines.append(f"Turn {utterance.turn}: {audio_summary}")
    return lines
