from __future__ import annotations

from dataclasses import dataclass


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _project_turn_sets(
    pair_sets: list[set[tuple[int, int]]],
    *,
    index: int,
) -> list[set[int]]:
    return [{pair[index] for pair in pair_set} for pair_set in pair_sets]


def _count_prf(
    gold_sets: list[set[int]],
    pred_sets: list[set[int]],
) -> tuple[int, int, int, float, float, float]:
    tp = sum(len(gold & pred) for gold, pred in zip(gold_sets, pred_sets))
    fp = sum(len(pred - gold) for gold, pred in zip(gold_sets, pred_sets))
    fn = sum(len(gold - pred) for gold, pred in zip(gold_sets, pred_sets))
    precision, recall, f1 = _prf(tp, fp, fn)
    return tp, fp, fn, precision, recall, f1


@dataclass(slots=True)
class BridgeProtSummary:
    num_dialogues: int
    parsed_dialogues: int
    strict_valid_dialogues: int
    total_records: int
    valid_records: int
    strict_pair_precision: float
    strict_pair_recall: float
    strict_pair_f1: float
    strict_emotion_turn_precision: float
    strict_emotion_turn_recall: float
    strict_emotion_turn_f1: float
    strict_cause_turn_precision: float
    strict_cause_turn_recall: float
    strict_cause_turn_f1: float
    salvaged_pair_precision: float
    salvaged_pair_recall: float
    salvaged_pair_f1: float
    salvaged_emotion_turn_precision: float
    salvaged_emotion_turn_recall: float
    salvaged_emotion_turn_f1: float
    salvaged_cause_turn_precision: float
    salvaged_cause_turn_recall: float
    salvaged_cause_turn_f1: float
    parsability: float
    strict_validity_rate: float
    valid_record_rate: float
    strict_tp: int
    strict_fp: int
    strict_fn: int
    strict_emotion_turn_tp: int
    strict_emotion_turn_fp: int
    strict_emotion_turn_fn: int
    strict_cause_turn_tp: int
    strict_cause_turn_fp: int
    strict_cause_turn_fn: int
    salvaged_tp: int
    salvaged_fp: int
    salvaged_fn: int
    salvaged_emotion_turn_tp: int
    salvaged_emotion_turn_fp: int
    salvaged_emotion_turn_fn: int
    salvaged_cause_turn_tp: int
    salvaged_cause_turn_fp: int
    salvaged_cause_turn_fn: int


def summarize_bridgeprot(
    *,
    gold_pair_sets: list[set[tuple[int, int]]],
    strict_pair_sets: list[set[tuple[int, int]]],
    salvaged_pair_sets: list[set[tuple[int, int]]],
    parsed_flags: list[bool],
    strict_valid_flags: list[bool],
    total_records: int,
    valid_records: int,
) -> BridgeProtSummary:
    strict_tp = sum(len(gold & pred) for gold, pred in zip(gold_pair_sets, strict_pair_sets))
    strict_fp = sum(len(pred - gold) for gold, pred in zip(gold_pair_sets, strict_pair_sets))
    strict_fn = sum(len(gold - pred) for gold, pred in zip(gold_pair_sets, strict_pair_sets))
    strict_p, strict_r, strict_f1 = _prf(strict_tp, strict_fp, strict_fn)

    gold_emotion_turn_sets = _project_turn_sets(gold_pair_sets, index=0)
    strict_emotion_turn_sets = _project_turn_sets(strict_pair_sets, index=0)
    (
        strict_emotion_turn_tp,
        strict_emotion_turn_fp,
        strict_emotion_turn_fn,
        strict_emotion_turn_p,
        strict_emotion_turn_r,
        strict_emotion_turn_f1,
    ) = _count_prf(gold_emotion_turn_sets, strict_emotion_turn_sets)

    gold_cause_turn_sets = _project_turn_sets(gold_pair_sets, index=1)
    strict_cause_turn_sets = _project_turn_sets(strict_pair_sets, index=1)
    (
        strict_cause_turn_tp,
        strict_cause_turn_fp,
        strict_cause_turn_fn,
        strict_cause_turn_p,
        strict_cause_turn_r,
        strict_cause_turn_f1,
    ) = _count_prf(gold_cause_turn_sets, strict_cause_turn_sets)

    salvaged_tp = sum(len(gold & pred) for gold, pred in zip(gold_pair_sets, salvaged_pair_sets))
    salvaged_fp = sum(len(pred - gold) for gold, pred in zip(gold_pair_sets, salvaged_pair_sets))
    salvaged_fn = sum(len(gold - pred) for gold, pred in zip(gold_pair_sets, salvaged_pair_sets))
    salvaged_p, salvaged_r, salvaged_f1 = _prf(salvaged_tp, salvaged_fp, salvaged_fn)

    salvaged_emotion_turn_sets = _project_turn_sets(salvaged_pair_sets, index=0)
    (
        salvaged_emotion_turn_tp,
        salvaged_emotion_turn_fp,
        salvaged_emotion_turn_fn,
        salvaged_emotion_turn_p,
        salvaged_emotion_turn_r,
        salvaged_emotion_turn_f1,
    ) = _count_prf(gold_emotion_turn_sets, salvaged_emotion_turn_sets)

    salvaged_cause_turn_sets = _project_turn_sets(salvaged_pair_sets, index=1)
    (
        salvaged_cause_turn_tp,
        salvaged_cause_turn_fp,
        salvaged_cause_turn_fn,
        salvaged_cause_turn_p,
        salvaged_cause_turn_r,
        salvaged_cause_turn_f1,
    ) = _count_prf(gold_cause_turn_sets, salvaged_cause_turn_sets)

    num_dialogues = len(gold_pair_sets)
    parsed_dialogues = sum(1 for flag in parsed_flags if flag)
    strict_valid_dialogues = sum(1 for flag in strict_valid_flags if flag)

    return BridgeProtSummary(
        num_dialogues=num_dialogues,
        parsed_dialogues=parsed_dialogues,
        strict_valid_dialogues=strict_valid_dialogues,
        total_records=total_records,
        valid_records=valid_records,
        strict_pair_precision=strict_p,
        strict_pair_recall=strict_r,
        strict_pair_f1=strict_f1,
        strict_emotion_turn_precision=strict_emotion_turn_p,
        strict_emotion_turn_recall=strict_emotion_turn_r,
        strict_emotion_turn_f1=strict_emotion_turn_f1,
        strict_cause_turn_precision=strict_cause_turn_p,
        strict_cause_turn_recall=strict_cause_turn_r,
        strict_cause_turn_f1=strict_cause_turn_f1,
        salvaged_pair_precision=salvaged_p,
        salvaged_pair_recall=salvaged_r,
        salvaged_pair_f1=salvaged_f1,
        salvaged_emotion_turn_precision=salvaged_emotion_turn_p,
        salvaged_emotion_turn_recall=salvaged_emotion_turn_r,
        salvaged_emotion_turn_f1=salvaged_emotion_turn_f1,
        salvaged_cause_turn_precision=salvaged_cause_turn_p,
        salvaged_cause_turn_recall=salvaged_cause_turn_r,
        salvaged_cause_turn_f1=salvaged_cause_turn_f1,
        parsability=parsed_dialogues / num_dialogues if num_dialogues > 0 else 0.0,
        strict_validity_rate=strict_valid_dialogues / num_dialogues if num_dialogues > 0 else 0.0,
        valid_record_rate=valid_records / total_records if total_records > 0 else 0.0,
        strict_tp=strict_tp,
        strict_fp=strict_fp,
        strict_fn=strict_fn,
        strict_emotion_turn_tp=strict_emotion_turn_tp,
        strict_emotion_turn_fp=strict_emotion_turn_fp,
        strict_emotion_turn_fn=strict_emotion_turn_fn,
        strict_cause_turn_tp=strict_cause_turn_tp,
        strict_cause_turn_fp=strict_cause_turn_fp,
        strict_cause_turn_fn=strict_cause_turn_fn,
        salvaged_tp=salvaged_tp,
        salvaged_fp=salvaged_fp,
        salvaged_fn=salvaged_fn,
        salvaged_emotion_turn_tp=salvaged_emotion_turn_tp,
        salvaged_emotion_turn_fp=salvaged_emotion_turn_fp,
        salvaged_emotion_turn_fn=salvaged_emotion_turn_fn,
        salvaged_cause_turn_tp=salvaged_cause_turn_tp,
        salvaged_cause_turn_fp=salvaged_cause_turn_fp,
        salvaged_cause_turn_fn=salvaged_cause_turn_fn,
    )
