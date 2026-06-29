from __future__ import annotations

from core.config import BridgeProtDataConfig, BridgeProtProtocolConfig
from core.modalities import load_bridgeprot_multimodal_split
from core.serializer import serialize_dialogue
from freeform.prompts import build_freeform_user_content
from freeform.targets import render_freeform_target_text
from retrieval.bank import BridgeProtDemoExample


def build_freeform_demo_bank(
    *,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    source_split: str,
    max_examples: int | None = None,
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
                target_json=render_freeform_target_text(
                    dialogue,
                    protocol=protocol_config,
                ),
                user_content=build_freeform_user_content(
                    dialogue,
                    serialized,
                    protocol=protocol_config,
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
