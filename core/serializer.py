from __future__ import annotations

from data.schema import Dialogue


def serialize_utterance_line(
    dialogue: Dialogue,
    turn_index: int,
    *,
    include_turn_id: bool = True,
    include_speaker: bool = True,
) -> str:
    utterance = dialogue.utterances[turn_index]
    segments: list[str] = []
    if include_turn_id:
        segments.append(f"[{utterance.turn}]")
    if include_speaker and utterance.speaker:
        segments.append(f"{utterance.speaker}:")
    segments.append(utterance.text.strip())
    return " ".join(segment for segment in segments if segment)


def serialize_dialogue(
    dialogue: Dialogue,
    *,
    include_turn_id: bool = True,
    include_speaker: bool = True,
) -> str:
    return "\n".join(
        serialize_utterance_line(
            dialogue,
            index,
            include_turn_id=include_turn_id,
            include_speaker=include_speaker,
        )
        for index, _ in enumerate(dialogue.utterances)
    )
