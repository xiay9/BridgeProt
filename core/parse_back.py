from __future__ import annotations

from core.schema import BridgeDialogueOutput


def dialogue_output_to_pair_set(output: BridgeDialogueOutput) -> set[tuple[int, int]]:
    return {
        (record.emotion_turn, cause_turn)
        for record in output.records
        for cause_turn in record.evidence
    }
