from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.hf_env import configure_hf_environment

configure_hf_environment()

from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtMethodConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    BridgeProtRetrievalConfig,
    BridgeProtRunConfig,
    bridgeprot_evidence_tag,
    bridgeprot_execution_mode,
    resolve_bridgeprot_stage_methods,
)
from core.config_loader import (
    bridgeprot_infer_config_path,
    load_yaml_config,
    resolve_input_path,
)
from core.inference_artifacts import resolve_inference_model_config_from_training_output
from evaluation.bridgeprot_telemetry import write_bridgeprot_task_summary
from utils.wandb import init_wandb_run

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BridgeProt with offline vLLM inference and parse-back evaluation."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a BridgeProt YAML config.")
    group.add_argument(
        "--dataset",
        "--datasets",
        dest="dataset",
        help="Shortcut dataset name, e.g. ecf / mec4 / mecad.",
    )
    parser.add_argument("--split", default=None, help="Optional split override, e.g. valid or test.")
    parser.add_argument("--run-name", default=None, help="Optional override for run.name.")
    parser.add_argument("--output-dir", default=None, help="Optional absolute output directory override.")
    parser.add_argument(
        "--stage1-method",
        default=None,
        help="Primary BridgeProt method, e.g. zeroshot / fewshot / lora / sft. Defaults to self stage2.",
    )
    parser.add_argument(
        "--stage2-method",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--use-audio", action="store_true", help="Enable audio summary evidence.")
    parser.add_argument(
        "--use-audio-summary",
        action="store_true",
        help="Enable audio summary evidence explicitly.",
    )
    parser.add_argument(
        "--use-video",
        action="store_true",
        help="Enable native video evidence.",
    )
    parser.add_argument(
        "--audio-summary-backend",
        default=None,
        help="Optional override for data.audio_summary_backend.",
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=None,
        help="Optional override for the stage-2 local window size.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Optional default override for model.model_name.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=None,
        help="Optional override for model.tokenizer_name.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        default=None,
        help="Optional default LoRA adapter path override.",
    )
    parser.add_argument(
        "--train-output-dir",
        default=None,
        help="Optional default training output directory used to resolve LoRA/SFT inference artifacts automatically.",
    )
    parser.add_argument(
        "--stage1-train-output-dir",
        default=None,
        help="Optional training output directory override for stage-1 method.",
    )
    parser.add_argument(
        "--stage2-train-output-dir",
        default=None,
        help="Optional training output directory override for stage-2 method.",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="Optional cap on the number of dialogues to run.",
    )
    return parser.parse_args()


def _resolve_input_path(path: str | Path) -> Path:
    return resolve_input_path(path, cwd=Path.cwd(), code_root=CODE_ROOT, repo_root=REPO_ROOT)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return _resolve_input_path(args.config)
    return bridgeprot_infer_config_path(dataset=args.dataset, configs_root=CODE_ROOT / "configs" / "bridgeprot")


def _make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _result_payload(
    *,
    run_timestamp: str,
    result,
) -> dict[str, Any]:
    return {
        "run_timestamp": run_timestamp,
        "runtime_info": result.runtime_info,
        "summary": asdict(result.summary),
    }


def _write_result_artifacts(
    *,
    output_dir: Path,
    run_timestamp: str,
    result,
) -> None:
    _write_json(output_dir / "result.json", _to_serializable(_result_payload(run_timestamp=run_timestamp, result=result)))
    _write_jsonl(
        output_dir / "predictions.jsonl",
        [_to_serializable(asdict(row)) for row in result.dialogue_results],
    )
    for artifact_name, artifact_payload in result.artifacts.items():
        artifact_path = output_dir / artifact_name
        if artifact_path.suffix == ".jsonl" and isinstance(artifact_payload, list):
            _write_jsonl(artifact_path, _to_serializable(artifact_payload))
        else:
            _write_json(artifact_path, _to_serializable(artifact_payload))


def _apply_default_model_overrides(
    *,
    base_model_config: BridgeProtModelConfig,
    model_name_or_path: str | None,
    tokenizer_name_or_path: str | None,
    lora_adapter_path: str | None,
) -> BridgeProtModelConfig:
    model_kwargs = asdict(base_model_config)
    if model_name_or_path is not None:
        model_kwargs["model_name"] = model_name_or_path
    if tokenizer_name_or_path is not None:
        model_kwargs["tokenizer_name"] = tokenizer_name_or_path
    if lora_adapter_path is not None:
        model_kwargs["lora_adapter_path"] = lora_adapter_path
    return BridgeProtModelConfig(**model_kwargs)


def _resolve_stage_model_config(
    *,
    method_name: str,
    base_model_config: BridgeProtModelConfig,
    data_config: BridgeProtDataConfig,
    train_output_dir: str | None,
) -> BridgeProtModelConfig:
    if method_name not in {"lora", "sft"}:
        return base_model_config
    if train_output_dir is None:
        return base_model_config
    return resolve_inference_model_config_from_training_output(
        train_output_dir=train_output_dir,
        base_model_config=base_model_config,
        use_video=data_config.use_video,
        expected_method=method_name,
    )


def main() -> None:
    args = _parse_args()
    run_timestamp = _make_run_timestamp()

    from methods.staged import run_bridgeprot_stage_pipeline
    from core.vllm_engine import detect_visible_gpu_count
    from utils.runtime import has_visible_cuda_gpu, set_random_seed

    if not has_visible_cuda_gpu():
        raise RuntimeError("CUDA is required for this script, but no visible NVIDIA GPU was detected.")

    config_path = _resolve_config_path(args)
    payload = load_yaml_config(config_path)

    run_kwargs = dict(payload.get("run", {}))
    if args.run_name is not None:
        run_kwargs["name"] = args.run_name
    run_cfg = BridgeProtRunConfig(**run_kwargs)

    dataset_name = payload["dataset_name"]
    data_kwargs = {"dataset_name": dataset_name, **payload.get("data", {})}
    if args.split is not None:
        data_kwargs["split"] = args.split
    if args.max_dialogues is not None:
        data_kwargs["max_dialogues"] = args.max_dialogues
    if args.use_audio:
        data_kwargs["use_audio"] = True
        data_kwargs["use_audio_summary"] = True
    if args.use_audio_summary:
        data_kwargs["use_audio_summary"] = True
    if args.use_video:
        data_kwargs["use_video"] = True
    if args.audio_summary_backend is not None:
        data_kwargs["audio_summary_backend"] = args.audio_summary_backend
    data_cfg = BridgeProtDataConfig(**data_kwargs)

    base_model_cfg = BridgeProtModelConfig(**payload.get("model", {}))
    default_model_cfg = _apply_default_model_overrides(
        base_model_config=base_model_cfg,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        lora_adapter_path=args.lora_adapter_path,
    )
    protocol_cfg = BridgeProtProtocolConfig(**payload.get("protocol", {}))
    decode_cfg = BridgeProtDecodeConfig(**payload.get("decode", {}))
    method_kwargs = dict(payload.get("method", {}))
    if args.stage2_method is not None and args.stage1_method is None:
        raise ValueError("--stage2-method requires --stage1-method.")
    if (
        args.stage1_method is not None
        and args.stage2_method is not None
        and args.stage1_method.lower() != args.stage2_method.lower()
    ):
        raise ValueError(
            "BridgeProt inference now supports self stage2 only. "
            "--stage2-method must match --stage1-method."
        )
    if args.stage1_method is not None:
        method_kwargs["stage1_method"] = args.stage1_method
        method_kwargs["name"] = args.stage1_method
        method_kwargs["stage2_method"] = args.stage1_method
        method_kwargs["stage"] = 2
    if args.stage2_method is not None:
        method_kwargs["stage2_method"] = args.stage2_method
        method_kwargs["stage"] = 2
    if args.candidate_window is not None:
        method_kwargs["candidate_window"] = args.candidate_window
    method_cfg = BridgeProtMethodConfig(**method_kwargs)
    stage, stage1_method, stage2_method = resolve_bridgeprot_stage_methods(method_cfg)
    retrieval_cfg = BridgeProtRetrievalConfig(**payload.get("retrieval", {}))
    execution_mode = bridgeprot_execution_mode(use_video=data_cfg.use_video)
    evidence_tag = bridgeprot_evidence_tag(
        use_audio=data_cfg.use_audio,
        use_video=data_cfg.use_video,
        use_audio_summary=data_cfg.audio_summary_enabled,
    )
    stage1_train_output_dir = args.stage1_train_output_dir or args.train_output_dir
    stage2_train_output_dir = args.stage2_train_output_dir or args.train_output_dir
    stage1_model_cfg = _resolve_stage_model_config(
        method_name=stage1_method,
        base_model_config=default_model_cfg,
        data_config=data_cfg,
        train_output_dir=stage1_train_output_dir,
    )
    stage2_model_cfg = _resolve_stage_model_config(
        method_name=stage2_method,
        base_model_config=default_model_cfg,
        data_config=data_cfg,
        train_output_dir=stage2_train_output_dir,
    )
    method_label = f"{stage1_method}-to-{stage2_method}"

    set_random_seed(run_cfg.seed, include_cuda=False)

    output_dir = Path(
        args.output_dir
        or run_cfg.resolve_output_dir(
            data_cfg.dataset_name,
            run_timestamp,
            method_name=method_label,
            stage=stage,
            split=data_cfg.split,
            output_mode=decode_cfg.output_mode,
            use_audio=data_cfg.use_audio,
            use_video=data_cfg.use_video,
            use_audio_summary=data_cfg.audio_summary_enabled,
            candidate_window=method_cfg.candidate_window,
            stage1_method=stage1_method,
            stage2_method=stage2_method,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "run": asdict(run_cfg),
        "run_timestamp": run_timestamp,
        "dataset_name": data_cfg.dataset_name,
        "data": asdict(data_cfg),
        "model": asdict(default_model_cfg),
        "stage_models": {
            "stage1": asdict(stage1_model_cfg),
            "stage2": asdict(stage2_model_cfg),
        },
        "protocol": asdict(protocol_cfg),
        "decode": asdict(decode_cfg),
        "method": asdict(method_cfg),
        "resolved_stage": {
            "stage": stage,
            "stage1_method": stage1_method,
            "stage2_method": stage2_method,
            "method_label": method_label,
        },
        "training_artifacts": {
            "stage1_train_output_dir": stage1_train_output_dir,
            "stage2_train_output_dir": stage2_train_output_dir,
        },
        "retrieval": asdict(retrieval_cfg),
        "resolved_mode": {
            "execution_mode": execution_mode,
            "evidence_tag": evidence_tag,
        },
        "runtime": {
            "device": "cuda",
            "num_visible_gpus": detect_visible_gpu_count(),
        },
    }
    _write_yaml(output_dir / "resolved_config.yaml", _to_serializable(resolved_config))

    print(
        f"run={run_cfg.name} dataset={data_cfg.dataset_name} split={data_cfg.split} "
        f"stage={stage} stage1_method={stage1_method} stage2_method={stage2_method} "
        f"device=cuda execution_mode={execution_mode}"
    )
    print(f"config={config_path}")
    print(f"output_dir={output_dir}")

    wandb_run = init_wandb_run(
        enabled=run_cfg.wandb_enabled,
        project=run_cfg.wandb_project,
        entity=run_cfg.wandb_entity,
        group=run_cfg.wandb_group
        or f"bridgeprot-stage{stage}-{stage1_method}-to-{stage2_method}-{data_cfg.dataset_name.lower()}",
        tags=[
            "bridgeprot",
            f"stage{stage}",
            f"stage1-{stage1_method}",
            f"stage2-{stage2_method}",
            data_cfg.dataset_name.lower(),
            data_cfg.split.lower(),
            *list(run_cfg.wandb_tags),
        ],
        run_name=run_cfg.name,
        run_timestamp=run_timestamp,
        output_dir=output_dir,
        config=_to_serializable(resolved_config),
    )
    if wandb_run is not None:
        print(f"wandb_url={wandb_run.url}")

    try:
        result = run_bridgeprot_stage_pipeline(
            data_config=data_cfg,
            protocol_config=protocol_cfg,
            decode_config=decode_cfg,
            retrieval_config=retrieval_cfg,
            method_config=method_cfg,
            stage1_model_config=stage1_model_cfg,
            stage2_model_config=stage2_model_cfg,
            seed=run_cfg.seed,
        )
        if stage == 2:
            stage1_result = result.stage_results.get("stage1")
            if stage1_result is None:
                raise RuntimeError("stage=2 completed without a stage1 result artifact.")
            _write_result_artifacts(
                output_dir=output_dir / "stage1",
                run_timestamp=run_timestamp,
                result=stage1_result,
            )
            _write_result_artifacts(
                output_dir=output_dir / "stage2",
                run_timestamp=run_timestamp,
                result=result,
            )
        else:
            _write_result_artifacts(
                output_dir=output_dir / "stage1",
                run_timestamp=run_timestamp,
                result=result,
            )

        write_bridgeprot_task_summary(
            output_dir=output_dir,
            kind="infer-stage2-self",
            task_metadata={
                "dataset": data_cfg.dataset_name.lower(),
                "split": data_cfg.split,
                "stage": stage,
                "stage1_method": stage1_method,
                "stage2_method": stage2_method,
                "run_name": run_cfg.name,
            },
        )

        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "output_dir": str(output_dir),
                    "num_dialogues": result.summary.num_dialogues,
                    "parsed_dialogues": result.summary.parsed_dialogues,
                    "strict_valid_dialogues": result.summary.strict_valid_dialogues,
                    "strict_pair_f1": result.summary.strict_pair_f1,
                    "strict_emotion_turn_f1": result.summary.strict_emotion_turn_f1,
                    "strict_cause_turn_f1": result.summary.strict_cause_turn_f1,
                    "salvaged_pair_f1": result.summary.salvaged_pair_f1,
                    "salvaged_emotion_turn_f1": result.summary.salvaged_emotion_turn_f1,
                    "salvaged_cause_turn_f1": result.summary.salvaged_cause_turn_f1,
                    "parsability": result.summary.parsability,
                    "valid_record_rate": result.summary.valid_record_rate,
                    "tensor_parallel_size": result.runtime_info["tensor_parallel_size"],
                    "num_visible_gpus": result.runtime_info["num_visible_gpus"],
                    "gpu_memory_utilization": result.runtime_info["gpu_memory_utilization"],
                    "min_free_gpu_memory_gb": result.runtime_info["min_free_gpu_memory_gb"],
                }
            )
            wandb_run.log(
                {
                    "summary/strict_pair_precision": result.summary.strict_pair_precision,
                    "summary/strict_pair_recall": result.summary.strict_pair_recall,
                    "summary/strict_pair_f1": result.summary.strict_pair_f1,
                    "summary/strict_emotion_turn_precision": result.summary.strict_emotion_turn_precision,
                    "summary/strict_emotion_turn_recall": result.summary.strict_emotion_turn_recall,
                    "summary/strict_emotion_turn_f1": result.summary.strict_emotion_turn_f1,
                    "summary/strict_cause_turn_precision": result.summary.strict_cause_turn_precision,
                    "summary/strict_cause_turn_recall": result.summary.strict_cause_turn_recall,
                    "summary/strict_cause_turn_f1": result.summary.strict_cause_turn_f1,
                    "summary/salvaged_pair_precision": result.summary.salvaged_pair_precision,
                    "summary/salvaged_pair_recall": result.summary.salvaged_pair_recall,
                    "summary/salvaged_pair_f1": result.summary.salvaged_pair_f1,
                    "summary/salvaged_emotion_turn_precision": result.summary.salvaged_emotion_turn_precision,
                    "summary/salvaged_emotion_turn_recall": result.summary.salvaged_emotion_turn_recall,
                    "summary/salvaged_emotion_turn_f1": result.summary.salvaged_emotion_turn_f1,
                    "summary/salvaged_cause_turn_precision": result.summary.salvaged_cause_turn_precision,
                    "summary/salvaged_cause_turn_recall": result.summary.salvaged_cause_turn_recall,
                    "summary/salvaged_cause_turn_f1": result.summary.salvaged_cause_turn_f1,
                    "summary/parsability": result.summary.parsability,
                    "summary/strict_validity_rate": result.summary.strict_validity_rate,
                    "summary/valid_record_rate": result.summary.valid_record_rate,
                }
            )
            wandb_run.finish()

        print(
            f"strict_pair_f1={result.summary.strict_pair_f1:.4f} "
            f"strict_emotion_turn_f1={result.summary.strict_emotion_turn_f1:.4f} "
            f"strict_cause_turn_f1={result.summary.strict_cause_turn_f1:.4f} "
            f"salvaged_pair_f1={result.summary.salvaged_pair_f1:.4f} "
            f"salvaged_emotion_turn_f1={result.summary.salvaged_emotion_turn_f1:.4f} "
            f"salvaged_cause_turn_f1={result.summary.salvaged_cause_turn_f1:.4f} "
            f"parsability={result.summary.parsability:.4f} "
            f"strict_validity_rate={result.summary.strict_validity_rate:.4f}"
        )
        print(
            f"num_visible_gpus={result.runtime_info['num_visible_gpus']} "
            f"tensor_parallel_size={result.runtime_info['tensor_parallel_size']} "
            f"gpu_memory_utilization={result.runtime_info['gpu_memory_utilization']:.2f} "
            f"min_free_gpu_memory_gb={result.runtime_info['min_free_gpu_memory_gb']:.2f}"
        )
        print(f"run_timestamp={run_timestamp}")
    except Exception:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
