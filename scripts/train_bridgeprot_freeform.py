from __future__ import annotations

import gc
from dataclasses import asdict
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

import argparse

from utils.hf_env import configure_hf_environment

configure_hf_environment()

from evaluation.bridgeprot_telemetry import write_bridgeprot_task_summary
from core.config_loader import (
    bridgeprot_train_config_path,
    load_yaml_config,
    resolve_input_path,
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BridgeProt free-form SFT or LoRA.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a BridgeProt training YAML config.")
    group.add_argument("--dataset", help="Shortcut dataset name, e.g. ecf / mec4 / mecad.")
    parser.add_argument("--method", default="sft", choices=["sft", "lora"], help="Training method: sft or lora.")
    parser.add_argument("--run-name", default=None, help="Optional override for training.run_name.")
    parser.add_argument("--output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-test-infer",
        action="store_true",
        help="Skip post-train free-form test inference.",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Split used for post-train inference. Defaults to test.",
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
    parser.add_argument("--use-video", action="store_true", help="Enable native video evidence.")
    parser.add_argument(
        "--audio-summary-backend",
        default=None,
        help="Optional override for data.audio_summary_backend.",
    )
    parser.add_argument(
        "--posthoc-extract",
        action="store_true",
        help="Apply deterministic post-hoc extraction during optional post-train test inference.",
    )
    parser.add_argument(
        "--test-method",
        choices=["zeroshot", "fewshot", "sft", "lora"],
        default=None,
        help="Optional method override for post-train free-form test inference. Defaults to the training method.",
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


def main() -> None:
    args = _parse_args()

    from core.config import (
        BridgeProtDataConfig,
        BridgeProtDecodeConfig,
        BridgeProtModelConfig,
        BridgeProtProtocolConfig,
        BridgeProtRetrievalConfig,
    )
    from core.inference_artifacts import resolve_inference_model_config_from_training_output
    from freeform.runner import run_bridgeprot_freeform
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
    retrieval_cfg = BridgeProtRetrievalConfig(**payload.get("retrieval", {}))
    training_kwargs = dict(payload.get("training", {}))
    training_kwargs["target_mode"] = "freeform"
    if args.run_name is not None:
        training_kwargs["run_name"] = args.run_name
    if args.output_dir is not None:
        training_kwargs["output_dir"] = args.output_dir
    if args.max_train_dialogues is not None:
        training_kwargs["max_train_dialogues"] = args.max_train_dialogues
    if args.max_eval_dialogues is not None:
        training_kwargs["max_eval_dialogues"] = args.max_eval_dialogues
    if args.num_train_epochs is not None:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
    training_cfg = BridgeProtTrainingConfig(**training_kwargs)

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
    _print_training_summary(training_result)

    resolved_config = {
        "dataset_name": data_cfg.dataset_name,
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "protocol": asdict(protocol_cfg),
        "retrieval": asdict(retrieval_cfg),
        "training": asdict(training_cfg),
        "freeform": {
            "posthoc_extract": args.posthoc_extract,
        },
        "runtime": {
            "config_path": str(config_path),
        },
    }
    _write_yaml(output_dir / "resolved_config.yaml", _to_serializable(resolved_config))

    if not args.skip_test_infer:
        test_data_cfg = BridgeProtDataConfig(
            dataset_name=data_cfg.dataset_name,
            split=args.test_split,
            data_root=data_cfg.data_root,
            max_dialogues=args.max_test_dialogues,
            include_speaker=data_cfg.include_speaker,
            include_turn_id=data_cfg.include_turn_id,
            use_audio=data_cfg.use_audio,
            use_video=data_cfg.use_video,
            use_audio_summary=data_cfg.use_audio_summary,
            audio_summary_backend=data_cfg.audio_summary_backend,
        )
        infer_model_cfg = resolve_inference_model_config_from_training_output(
            train_output_dir=output_dir,
            base_model_config=model_cfg,
            use_video=data_cfg.use_video,
            expected_method=training_cfg.method.lower(),
        )
        decode_cfg = BridgeProtDecodeConfig(output_mode="freeform")
        infer_method = (args.test_method or training_cfg.method).lower()

        _release_cuda_memory()
        test_result = run_bridgeprot_freeform(
            data_config=test_data_cfg,
            model_config=infer_model_cfg,
            protocol_config=protocol_cfg,
            decode_config=decode_cfg,
            seed=training_cfg.seed,
            method_name=infer_method,
            retrieval_config=retrieval_cfg,
            posthoc_extract=args.posthoc_extract,
        )
        _write_stage_result_artifacts(
            output_dir=output_dir / "stage1",
            result=test_result,
        )
        _write_json(
            output_dir / "test_result.json",
            _to_serializable(
                {
                    "split": test_data_cfg.split,
                    "runtime_info": test_result.runtime_info,
                    "summary": asdict(test_result.summary),
                    "model": asdict(infer_model_cfg),
                    "method": infer_method,
                    "posthoc_extract": args.posthoc_extract,
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
            f"salvaged_pair_f1={test_result.summary.salvaged_pair_f1:.4f} "
            f"parsability={test_result.summary.parsability:.4f} "
            f"strict_validity_rate={test_result.summary.strict_validity_rate:.4f}"
        )

    write_bridgeprot_task_summary(
        output_dir=output_dir,
        kind="train",
        task_metadata={
            "dataset": data_cfg.dataset_name.lower(),
            "method": training_cfg.method.lower(),
            "run_name": training_cfg.run_name,
            "protocol": "freeform",
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
