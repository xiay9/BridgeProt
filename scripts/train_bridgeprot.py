from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils.hf_env import configure_hf_environment

configure_hf_environment()

from evaluation.bridgeprot_telemetry import write_bridgeprot_task_summary
from core.config_loader import (
    bridgeprot_train_config_path,
    load_yaml_config,
    resolve_input_path,
)
AUTO_DDP_MARKER_ENV = "BRIDGEPROT_AUTO_DDP_OUTPUT_MARKER"
AUTO_DDP_OUTPUT_DIR_ENV = "BRIDGEPROT_AUTO_DDP_OUTPUT_DIR"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BridgeProt with SFT or LoRA.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a BridgeProt training YAML config.")
    group.add_argument("--dataset", help="Shortcut dataset name, e.g. ecf / mec4 / mecad.")
    parser.add_argument("--method", default="sft", help="Training method: sft or lora.")
    parser.add_argument(
        "--target-mode",
        choices=["minimal", "full"],
        default=None,
        help="Training target JSON mode. Defaults to config value; minimal is recommended.",
    )
    parser.add_argument("--run-name", default=None, help="Optional override for training.run_name.")
    parser.add_argument("--output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-test-infer",
        action="store_true",
        help="Skip post-train BridgeProt test inference.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Split used for post-train inference. Defaults to test.",
    )
    parser.add_argument(
        "--eval-stage",
        type=int,
        choices=[2],
        default=2,
        help="Evaluation stage after training. Only stage=2 is supported.",
    )
    parser.add_argument(
        "--max-test-dialogues",
        type=int,
        default=None,
        help="Optional cap on the number of test dialogues for post-train inference.",
    )
    parser.add_argument(
        "--max-train-dialogues",
        type=int,
        default=None,
        help="Optional cap on the number of train dialogues used for supervised training.",
    )
    parser.add_argument(
        "--max-eval-dialogues",
        type=int,
        default=None,
        help="Optional cap on the number of valid dialogues used for supervised eval during training.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="Optional override for training.num_train_epochs.",
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
        "--distributed-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _resolve_input_path(path: str | Path) -> Path:
    return resolve_input_path(path, cwd=Path.cwd(), code_root=CODE_ROOT, repo_root=REPO_ROOT)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return _resolve_input_path(args.config)
    return bridgeprot_train_config_path(
        dataset=args.dataset,
        method=args.method,
        configs_root=CODE_ROOT / "configs" / "bridgeprot",
    )


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_stage_result_artifacts(
    *,
    output_dir: Path,
    result,
) -> None:
    _write_json(
        output_dir / "result.json",
        _to_serializable(
            {
                "runtime_info": result.runtime_info,
                "summary": asdict(result.summary),
            }
        ),
    )
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


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_training_summary(training_result) -> None:
    print(
        f"train_loss={_format_metric(training_result.train_metrics.get('train_loss'))} "
        f"train_runtime={_format_metric(training_result.train_metrics.get('train_runtime'))} "
        f"global_step={training_result.global_step}"
    )
    print(
        f"best_eval_loss={_format_metric(training_result.best_eval_loss)} "
        f"final_eval_loss={_format_metric((training_result.final_eval_metrics or {}).get('eval_loss'))} "
        f"best_model_checkpoint={training_result.best_model_checkpoint or 'n/a'}"
    )


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_distributed_context() -> bool:
    return _distributed_world_size() > 1 or "LOCAL_RANK" in os.environ


def _is_world_process_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _visible_gpu_ids_from_env() -> list[int] | None:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return None

    gpu_ids: list[int] = []
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        gpu_ids.append(int(token))
    return gpu_ids or None


def _detect_visible_gpu_ids() -> list[int]:
    visible_gpu_ids = _visible_gpu_ids_from_env()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            import torch
        except ModuleNotFoundError:
            return []
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        all_gpu_ids = list(range(count))
        if visible_gpu_ids is not None:
            return [gpu_id for gpu_id in visible_gpu_ids if gpu_id in all_gpu_ids]
        return all_gpu_ids

    visible_set = set(visible_gpu_ids) if visible_gpu_ids is not None else None
    candidate_ids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            gpu_id = int(line.strip())
        except ValueError:
            continue
        if visible_set is not None and gpu_id not in visible_set:
            continue
        candidate_ids.append(gpu_id)
    return candidate_ids


def _load_training_result_json(output_dir: Path):
    from methods.training import BridgeProtTrainingResult

    payload = json.loads((output_dir / "train_result.json").read_text(encoding="utf-8"))
    return BridgeProtTrainingResult(
        output_dir=Path(payload["output_dir"]),
        global_step=int(payload["global_step"]),
        train_metrics=dict(payload["train_metrics"]),
        final_eval_metrics=payload.get("final_eval_metrics"),
        best_model_checkpoint=payload.get("best_model_checkpoint"),
        best_eval_loss=payload.get("best_eval_loss"),
        merged_model_dir=payload.get("merged_model_dir"),
    )


def _write_output_marker(output_dir: Path) -> None:
    marker_path = os.environ.get(AUTO_DDP_MARKER_ENV)
    if not marker_path or not _is_world_process_zero():
        return
    Path(marker_path).write_text(str(output_dir), encoding="utf-8")


def _maybe_auto_launch_distributed(
    args: argparse.Namespace,
    *,
    data_cfg,
    training_cfg,
) -> Path | None:
    if args.distributed_worker or _is_distributed_context():
        return None

    visible_gpu_ids = _visible_gpu_ids_from_env()
    if visible_gpu_ids is not None:
        gpu_ids = sorted(visible_gpu_ids)
    else:
        gpu_ids = _detect_visible_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("Could not find any visible CUDA GPU for BridgeProt training.")

    gpu_ids = sorted(gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    if len(gpu_ids) == 1:
        print(f"auto_train_gpus={gpu_ids}")
        return None

    shared_output_dir = training_cfg.resolve_output_dir(
        data_cfg.dataset_name,
        use_audio=data_cfg.use_audio,
        use_video=data_cfg.use_video,
        use_audio_summary=data_cfg.audio_summary_enabled,
    )
    marker_fd, marker_path = tempfile.mkstemp(prefix="bridgeprot_train_", suffix=".marker")
    os.close(marker_fd)
    env = os.environ.copy()
    env[AUTO_DDP_MARKER_ENV] = marker_path
    env[AUTO_DDP_OUTPUT_DIR_ENV] = str(shared_output_dir)
    forwarded_args = [arg for arg in sys.argv[1:] if arg != "--distributed-worker"]
    if "--output-dir" not in forwarded_args:
        forwarded_args.extend(["--output-dir", str(shared_output_dir)])
    if "--skip-test-infer" not in forwarded_args:
        forwarded_args.append("--skip-test-infer")
    forwarded_args.append("--distributed-worker")

    print(f"auto_train_gpus={gpu_ids} launch=ddp")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(Path(__file__).resolve()),
        *forwarded_args,
    ]
    completed = subprocess.run(command, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    marker = Path(marker_path)
    if not marker.exists():
        raise RuntimeError("Distributed training finished but did not report an output directory.")
    output_dir = Path(marker.read_text(encoding="utf-8").strip())
    marker.unlink(missing_ok=True)
    return output_dir


def main() -> None:
    args = _parse_args()

    from core.config import (
        BridgeProtDataConfig,
        BridgeProtDecodeConfig,
        BridgeProtModelConfig,
        BridgeProtProtocolConfig,
    )
    from methods.training import BridgeProtTrainingConfig, train_bridgeprot_supervised
    from utils.runtime import set_random_seed

    config_path = _resolve_config_path(args)
    payload = load_yaml_config(config_path)

    data_cfg = BridgeProtDataConfig(
        dataset_name=payload["dataset_name"],
        **{
            **payload.get("data", {}),
            **({"use_audio": True} if args.use_audio else {}),
            **({"use_audio_summary": True} if (args.use_audio or args.use_audio_summary) else {}),
            **({"use_video": True} if args.use_video else {}),
            **({"audio_summary_backend": args.audio_summary_backend} if args.audio_summary_backend is not None else {}),
        },
    )
    model_cfg = BridgeProtModelConfig(**payload.get("model", {}))
    protocol_cfg = BridgeProtProtocolConfig(**payload.get("protocol", {}))
    training_kwargs = dict(payload.get("training", {}))
    if args.run_name is not None:
        training_kwargs["run_name"] = args.run_name
    if args.output_dir is not None:
        training_kwargs["output_dir"] = args.output_dir
    elif AUTO_DDP_OUTPUT_DIR_ENV in os.environ:
        training_kwargs["output_dir"] = os.environ[AUTO_DDP_OUTPUT_DIR_ENV]
    if args.target_mode is not None:
        training_kwargs["target_mode"] = args.target_mode
    if args.max_train_dialogues is not None:
        training_kwargs["max_train_dialogues"] = args.max_train_dialogues
    if args.max_eval_dialogues is not None:
        training_kwargs["max_eval_dialogues"] = args.max_eval_dialogues
    if args.num_train_epochs is not None:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
    training_cfg = BridgeProtTrainingConfig(**training_kwargs)
    auto_trained_output_dir = _maybe_auto_launch_distributed(
        args,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
    )

    if auto_trained_output_dir is None:
        set_random_seed(training_cfg.seed)
        training_result = train_bridgeprot_supervised(
            model_name=model_cfg.model_name,
            tokenizer_name=model_cfg.tokenizer_name,
            trust_remote_code=model_cfg.trust_remote_code,
            video_num_frames=model_cfg.video_num_frames,
            video_max_edge=model_cfg.video_max_edge,
            data_config=data_cfg,
            protocol_config=protocol_cfg,
            training_config=training_cfg,
        )
        output_dir = training_result.output_dir
        _write_output_marker(output_dir)

        if _distributed_world_size() > 1:
            return
        _print_training_summary(training_result)
    else:
        output_dir = auto_trained_output_dir
        training_result = _load_training_result_json(output_dir)
        _print_training_summary(training_result)

    resolved_config = {
        "dataset_name": data_cfg.dataset_name,
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "protocol": asdict(protocol_cfg),
        "training": asdict(training_cfg),
        "retrieval": payload.get("retrieval", {}),
        "runtime": {
            "config_path": str(config_path),
        },
    }
    _write_yaml(output_dir / "resolved_config.yaml", _to_serializable(resolved_config))

    if not args.skip_test_infer:
        from core.config import BridgeProtMethodConfig, BridgeProtRetrievalConfig
        from core.inference_artifacts import resolve_inference_model_config_from_training_output
        from methods.staged import run_bridgeprot_stage_pipeline

        test_data_cfg = BridgeProtDataConfig(
            dataset_name=data_cfg.dataset_name,
            split=args.test_split,
            data_root=data_cfg.data_root,
            max_dialogues=args.max_test_dialogues,
            include_speaker=data_cfg.include_speaker,
            include_turn_id=data_cfg.include_turn_id,
            use_audio=data_cfg.use_audio,
            use_video=data_cfg.use_video,
            audio_summary_backend=data_cfg.audio_summary_backend,
            use_audio_summary=data_cfg.use_audio_summary,
        )
        infer_model_cfg = resolve_inference_model_config_from_training_output(
            train_output_dir=output_dir,
            base_model_config=model_cfg,
            use_video=data_cfg.use_video,
            expected_method=training_cfg.method.lower(),
        )
        decode_cfg = BridgeProtDecodeConfig(output_mode=training_cfg.target_mode)
        method_cfg = BridgeProtMethodConfig(
            name=training_cfg.method.lower(),
            stage=args.eval_stage,
            stage1_method=training_cfg.method.lower(),
            stage2_method=training_cfg.method.lower(),
        )
        retrieval_cfg = BridgeProtRetrievalConfig(**payload.get("retrieval", {}))

        _release_cuda_memory()
        test_result = run_bridgeprot_stage_pipeline(
            data_config=test_data_cfg,
            protocol_config=protocol_cfg,
            decode_config=decode_cfg,
            retrieval_config=retrieval_cfg,
            method_config=method_cfg,
            stage1_model_config=infer_model_cfg,
            stage2_model_config=infer_model_cfg,
            seed=training_cfg.seed,
        )
        _write_stage_result_artifacts(
            output_dir=output_dir / "stage1",
            result=test_result.stage_results.get("stage1", test_result),
        )
        _write_stage_result_artifacts(
            output_dir=output_dir / "stage2",
            result=test_result,
        )
        _write_json(
            output_dir / "test_result.json",
            _to_serializable(
                {
                    "split": test_data_cfg.split,
                    "stage": args.eval_stage,
                    "runtime_info": test_result.runtime_info,
                    "summary": asdict(test_result.summary),
                    "model": asdict(infer_model_cfg),
                }
            ),
        )
        _write_jsonl(
            output_dir / "test_predictions.jsonl",
            [_to_serializable(asdict(row)) for row in test_result.dialogue_results],
        )
        print(
            f"test_split={test_data_cfg.split} "
            f"strict_pair_f1={test_result.summary.strict_pair_f1:.4f} "
            f"strict_emotion_turn_f1={test_result.summary.strict_emotion_turn_f1:.4f} "
            f"strict_cause_turn_f1={test_result.summary.strict_cause_turn_f1:.4f} "
            f"salvaged_pair_f1={test_result.summary.salvaged_pair_f1:.4f} "
            f"salvaged_emotion_turn_f1={test_result.summary.salvaged_emotion_turn_f1:.4f} "
            f"salvaged_cause_turn_f1={test_result.summary.salvaged_cause_turn_f1:.4f} "
            f"parsability={test_result.summary.parsability:.4f} "
            f"strict_validity_rate={test_result.summary.strict_validity_rate:.4f}"
        )
        print(
            f"test_result_json={output_dir / 'test_result.json'} "
            f"test_predictions_jsonl={output_dir / 'test_predictions.jsonl'}"
        )

    write_bridgeprot_task_summary(
        output_dir=output_dir,
        kind="train",
        task_metadata={
            "dataset": data_cfg.dataset_name.lower(),
            "method": training_cfg.method.lower(),
            "run_name": training_cfg.run_name,
            "eval_stage": None if args.skip_test_infer else args.eval_stage,
            "test_split": None if args.skip_test_infer else args.test_split,
        },
    )

    print(
        f"method={training_cfg.method} dataset={data_cfg.dataset_name} "
        f"model={model_cfg.model_name} output_dir={output_dir}"
    )
    print(f"config={config_path}")


if __name__ == "__main__":
    main()
