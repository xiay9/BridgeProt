from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else None


def _round_or_none(value: object, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _normalize_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _clean_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested = _clean_mapping(value)
            if nested:
                cleaned[key] = nested
            continue
        if isinstance(value, list):
            cleaned[key] = value
            continue
        cleaned[key] = value
    return cleaned


def _metric_triplet(
    *,
    prefix: str,
    summary: Mapping[str, Any],
    include_label: str,
) -> dict[str, Any]:
    return {
        "label": include_label,
        "pair": {
            "precision": _round_or_none(summary.get(f"{prefix}_pair_precision")),
            "recall": _round_or_none(summary.get(f"{prefix}_pair_recall")),
            "f1": _round_or_none(summary.get(f"{prefix}_pair_f1")),
            "tp": _int_or_none(summary.get(f"{prefix}_tp")),
            "fp": _int_or_none(summary.get(f"{prefix}_fp")),
            "fn": _int_or_none(summary.get(f"{prefix}_fn")),
        },
        "emotion_turn": {
            "precision": _round_or_none(summary.get(f"{prefix}_emotion_turn_precision")),
            "recall": _round_or_none(summary.get(f"{prefix}_emotion_turn_recall")),
            "f1": _round_or_none(summary.get(f"{prefix}_emotion_turn_f1")),
            "tp": _int_or_none(summary.get(f"{prefix}_emotion_turn_tp")),
            "fp": _int_or_none(summary.get(f"{prefix}_emotion_turn_fp")),
            "fn": _int_or_none(summary.get(f"{prefix}_emotion_turn_fn")),
        },
        "cause_turn": {
            "precision": _round_or_none(summary.get(f"{prefix}_cause_turn_precision")),
            "recall": _round_or_none(summary.get(f"{prefix}_cause_turn_recall")),
            "f1": _round_or_none(summary.get(f"{prefix}_cause_turn_f1")),
            "tp": _int_or_none(summary.get(f"{prefix}_cause_turn_tp")),
            "fp": _int_or_none(summary.get(f"{prefix}_cause_turn_fp")),
            "fn": _int_or_none(summary.get(f"{prefix}_cause_turn_fn")),
        },
    }


def _result_section(result_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not result_payload:
        return None
    summary = result_payload.get("summary")
    if not isinstance(summary, Mapping):
        return None
    runtime_info = result_payload.get("runtime_info")
    runtime_mapping = runtime_info if isinstance(runtime_info, Mapping) else {}
    section = {
        "num_dialogues": _int_or_none(summary.get("num_dialogues")),
        "parsed_dialogues": _int_or_none(summary.get("parsed_dialogues")),
        "strict_valid_dialogues": _int_or_none(summary.get("strict_valid_dialogues")),
        "total_records": _int_or_none(summary.get("total_records")),
        "valid_records": _int_or_none(summary.get("valid_records")),
        "parsability": _round_or_none(summary.get("parsability")),
        "strict_validity_rate": _round_or_none(summary.get("strict_validity_rate")),
        "valid_record_rate": _round_or_none(summary.get("valid_record_rate")),
        "strict": _metric_triplet(prefix="strict", summary=summary, include_label="strict"),
        "salvaged": _metric_triplet(prefix="salvaged", summary=summary, include_label="salvaged"),
        "performance": _clean_mapping(
            {
                "total_duration_sec": _round_or_none(runtime_mapping.get("total_duration_sec")),
                "stage1_duration_sec": _round_or_none(runtime_mapping.get("stage1_duration_sec")),
                "stage2_duration_sec": _round_or_none(runtime_mapping.get("stage2_duration_sec")),
                "dialogues_per_sec": _round_or_none(runtime_mapping.get("dialogues_per_sec")),
                "stage1_dialogues_per_sec": _round_or_none(runtime_mapping.get("stage1_dialogues_per_sec")),
                "stage2_dialogues_per_sec": _round_or_none(runtime_mapping.get("stage2_dialogues_per_sec")),
                "stage2_windows_per_sec": _round_or_none(runtime_mapping.get("stage2_windows_per_sec")),
                "num_windows": _int_or_none(runtime_mapping.get("num_windows")),
            }
        ),
        "runtime_info": dict(runtime_mapping),
    }
    return _clean_mapping(section)


def _extract_infer_summary(output_dir: Path, kind: str) -> dict[str, Any]:
    stage1_result = _load_json(output_dir / "stage1" / "result.json")
    stage2_result = _load_json(output_dir / "stage2" / "result.json")
    resolved_config = _load_yaml(output_dir / "resolved_config.yaml")

    stage1_section = _result_section(stage1_result)
    stage2_section = _result_section(stage2_result)
    performance: dict[str, Any] = {}
    if stage1_section:
        stage1_performance = stage1_section.get("performance")
        if isinstance(stage1_performance, Mapping):
            performance.update(stage1_performance)
    if stage2_section:
        stage2_performance = stage2_section.get("performance")
        if isinstance(stage2_performance, Mapping):
            performance.update(stage2_performance)

    stage2_delta_pair_f1 = None
    if stage1_section and stage2_section:
        stage1_f1 = (
            stage1_section.get("strict", {})
            .get("pair", {})
            .get("f1")
        )
        stage2_f1 = (
            stage2_section.get("strict", {})
            .get("pair", {})
            .get("f1")
        )
        if isinstance(stage1_f1, (int, float)) and isinstance(stage2_f1, (int, float)):
            stage2_delta_pair_f1 = round(float(stage2_f1) - float(stage1_f1), 4)

    return _clean_mapping(
        {
            "schema_version": "bridgeprot_task_summary_v1",
            "kind": kind,
            "paths": {
                "output_dir": str(output_dir),
                "config_snapshot": _normalize_path(output_dir / "resolved_config.yaml"),
                "stage1_result_json": _normalize_path(output_dir / "stage1" / "result.json"),
                "stage1_predictions_jsonl": _normalize_path(output_dir / "stage1" / "predictions.jsonl"),
                "stage2_result_json": _normalize_path(output_dir / "stage2" / "result.json")
                if (output_dir / "stage2" / "result.json").exists()
                else None,
                "stage2_predictions_jsonl": _normalize_path(output_dir / "stage2" / "predictions.jsonl")
                if (output_dir / "stage2" / "predictions.jsonl").exists()
                else None,
            },
            "config": {
                "dataset_name": ((resolved_config or {}).get("dataset_name")),
                "split": (((resolved_config or {}).get("data") or {}).get("split")),
                "run_name": (((resolved_config or {}).get("run") or {}).get("name")),
                "stage": ((((resolved_config or {}).get("resolved_stage") or {}).get("stage"))),
                "stage1_method": ((((resolved_config or {}).get("resolved_stage") or {}).get("stage1_method"))),
                "stage2_method": ((((resolved_config or {}).get("resolved_stage") or {}).get("stage2_method"))),
                "model_name": (((resolved_config or {}).get("model") or {}).get("model_name")),
                "use_audio": (((resolved_config or {}).get("data") or {}).get("use_audio")),
                "use_video": (((resolved_config or {}).get("data") or {}).get("use_video")),
                "use_audio_summary": (((resolved_config or {}).get("data") or {}).get("use_audio_summary")),
                "audio_summary_backend": (((resolved_config or {}).get("data") or {}).get("audio_summary_backend")),
            },
            "performance": performance,
            "correctness": {
                "stage1": stage1_section,
                "stage2": stage2_section,
                "stage2_delta_strict_pair_f1": stage2_delta_pair_f1,
            },
        }
    )


def _extract_train_summary(output_dir: Path) -> dict[str, Any]:
    train_result = _load_json(output_dir / "train_result.json") or {}
    train_config = _load_json(output_dir / "train_config.json") or {}
    resolved_config = _load_yaml(output_dir / "resolved_config.yaml")
    stage1_result = _load_json(output_dir / "stage1" / "result.json")
    stage2_result = _load_json(output_dir / "stage2" / "result.json")
    test_result = _load_json(output_dir / "test_result.json") or {}

    train_metrics = train_result.get("train_metrics")
    train_metrics = train_metrics if isinstance(train_metrics, Mapping) else {}
    final_eval_metrics = train_result.get("final_eval_metrics")
    final_eval_metrics = final_eval_metrics if isinstance(final_eval_metrics, Mapping) else {}

    performance = {
        "train_runtime_sec": _round_or_none(train_metrics.get("train_runtime")),
        "train_samples_per_second": _round_or_none(train_metrics.get("train_samples_per_second")),
        "train_steps_per_second": _round_or_none(train_metrics.get("train_steps_per_second")),
        "eval_runtime_sec": _round_or_none(final_eval_metrics.get("eval_runtime")),
        "eval_samples_per_second": _round_or_none(final_eval_metrics.get("eval_samples_per_second")),
        "eval_steps_per_second": _round_or_none(final_eval_metrics.get("eval_steps_per_second")),
        "global_step": _int_or_none(train_result.get("global_step")),
        "best_eval_loss": _round_or_none(train_result.get("best_eval_loss"), digits=6),
        "final_eval_loss": _round_or_none(final_eval_metrics.get("eval_loss"), digits=6),
    }

    return _clean_mapping(
        {
            "schema_version": "bridgeprot_task_summary_v1",
            "kind": "train",
            "paths": {
                "output_dir": str(output_dir),
                "config_snapshot": _normalize_path(output_dir / "resolved_config.yaml")
                if (output_dir / "resolved_config.yaml").exists()
                else _normalize_path(output_dir / "train_config.json"),
                "train_config_json": _normalize_path(output_dir / "train_config.json"),
                "train_result_json": _normalize_path(output_dir / "train_result.json"),
                "best_model_checkpoint": _normalize_path(train_result.get("best_model_checkpoint")),
                "merged_model_dir": _normalize_path(train_result.get("merged_model_dir")),
                "stage1_result_json": _normalize_path(output_dir / "stage1" / "result.json")
                if (output_dir / "stage1" / "result.json").exists()
                else None,
                "stage2_result_json": _normalize_path(output_dir / "stage2" / "result.json")
                if (output_dir / "stage2" / "result.json").exists()
                else None,
                "test_result_json": _normalize_path(output_dir / "test_result.json")
                if (output_dir / "test_result.json").exists()
                else None,
            },
            "config": {
                "method": ((resolved_config or {}).get("training") or train_config).get("method"),
                "dataset_name": ((resolved_config or {}).get("dataset_name")),
                "model_name": (((resolved_config or {}).get("model") or {}).get("model_name")),
                "target_mode": ((resolved_config or {}).get("training") or train_config).get("target_mode"),
                "use_audio": (((resolved_config or {}).get("data") or {}).get("use_audio")),
                "use_video": (((resolved_config or {}).get("data") or {}).get("use_video")),
                "use_audio_summary": (((resolved_config or {}).get("data") or {}).get("use_audio_summary")),
                "audio_summary_backend": (((resolved_config or {}).get("data") or {}).get("audio_summary_backend")),
            },
            "performance": performance,
            "correctness": {
                "stage1": _result_section(stage1_result),
                "stage2": _result_section(stage2_result),
                "test": _result_section(test_result),
            },
        }
    )


def build_bridgeprot_task_summary(
    *,
    output_dir: str | Path,
    kind: str,
    task_metadata: Mapping[str, Any] | None = None,
    resource_metrics: Mapping[str, Any] | None = None,
    duration_sec: float | None = None,
    returncode: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).resolve()
    if kind == "train":
        summary = _extract_train_summary(output_path)
    else:
        summary = _extract_infer_summary(output_path, kind=kind)

    summary["task"] = _clean_mapping(dict(task_metadata or {}))
    summary["resources"] = _clean_mapping(dict(resource_metrics or {}))
    execution = {
        "returncode": returncode,
        "task_duration_sec": _round_or_none(duration_sec, digits=1),
    }
    summary["execution"] = _clean_mapping(execution)
    return _clean_mapping(summary)


def write_bridgeprot_task_summary_payload(
    *,
    output_dir: str | Path,
    summary: Mapping[str, Any],
    filename: str = "task_summary.json",
) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / filename
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, ensure_ascii=False)
    return summary_path


def write_bridgeprot_task_summary(
    *,
    output_dir: str | Path,
    kind: str,
    task_metadata: Mapping[str, Any] | None = None,
    resource_metrics: Mapping[str, Any] | None = None,
    duration_sec: float | None = None,
    returncode: int | None = None,
    filename: str = "task_summary.json",
) -> Path:
    summary = build_bridgeprot_task_summary(
        output_dir=output_dir,
        kind=kind,
        task_metadata=task_metadata,
        resource_metrics=resource_metrics,
        duration_sec=duration_sec,
        returncode=returncode,
    )
    return write_bridgeprot_task_summary_payload(
        output_dir=output_dir,
        summary=summary,
        filename=filename,
    )
