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

from configs.paths import OUTPUT_ROOT
from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    BridgeProtRetrievalConfig,
)
from core.config_loader import (
    bridgeprot_infer_config_path,
    load_yaml_config,
    resolve_input_path,
)
from core.inference_artifacts import resolve_inference_model_config_from_training_output
from evaluation.bridgeprot_telemetry import write_bridgeprot_task_summary

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BridgeProt free-form single-pass inference with optional deterministic post-hoc extraction."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a BridgeProt YAML config.")
    group.add_argument("--dataset", help="Shortcut dataset name, e.g. ecf / mec4 / mecad.")
    parser.add_argument(
        "--method",
        choices=["zeroshot", "fewshot", "sft", "lora"],
        required=True,
        help="Free-form inference method.",
    )
    parser.add_argument("--split", default=None, help="Optional split override, e.g. valid or test.")
    parser.add_argument("--run-name", default=None, help="Optional override for run.name.")
    parser.add_argument("--output-dir", default=None, help="Optional absolute output directory override.")
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
        help="Optional training output directory used to resolve SFT/LoRA inference artifacts automatically.",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=None,
        help="Optional cap on the number of dialogues to run.",
    )
    parser.add_argument(
        "--posthoc-extract",
        action="store_true",
        help="Apply deterministic turn-index extraction to free-form outputs before scoring.",
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


def _write_result_artifacts(
    *,
    output_dir: Path,
    run_timestamp: str,
    result,
) -> None:
    _write_json(
        output_dir / "result.json",
        _to_serializable(
            {
                "run_timestamp": run_timestamp,
                "runtime_info": result.runtime_info,
                "summary": asdict(result.summary),
            }
        ),
    )
    _write_jsonl(
        output_dir / "predictions.jsonl",
        [_to_serializable(asdict(row)) for row in result.dialogue_results],
    )


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


def _resolve_method_model_config(
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

    from core.vllm_engine import detect_visible_gpu_count
    from freeform.runner import run_bridgeprot_freeform
    from utils.runtime import has_visible_cuda_gpu, set_random_seed

    if not has_visible_cuda_gpu():
        raise RuntimeError("CUDA is required for this script, but no visible NVIDIA GPU was detected.")

    config_path = _resolve_config_path(args)
    payload = load_yaml_config(config_path)

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
    model_cfg = _resolve_method_model_config(
        method_name=args.method.lower(),
        base_model_config=default_model_cfg,
        data_config=data_cfg,
        train_output_dir=args.train_output_dir,
    )
    protocol_cfg = BridgeProtProtocolConfig(**payload.get("protocol", {}))
    retrieval_cfg = BridgeProtRetrievalConfig(**payload.get("retrieval", {}))
    decode_kwargs = dict(payload.get("decode", {}))
    decode_kwargs["output_mode"] = "freeform"
    decode_cfg = BridgeProtDecodeConfig(**decode_kwargs)

    set_random_seed(42, include_cuda=False)

    output_dir = Path(args.output_dir or payload.get("run", {}).get("output_dir") or Path(
        OUTPUT_ROOT,
        "bridgeprot",
        "matrix-freeform",
        "manual",
        data_cfg.dataset_name.lower(),
        data_cfg.split.lower(),
        args.method.lower(),
        run_timestamp,
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "run_timestamp": run_timestamp,
        "dataset_name": data_cfg.dataset_name,
        "data": asdict(data_cfg),
        "model": asdict(default_model_cfg),
        "resolved_model": asdict(model_cfg),
        "protocol": asdict(protocol_cfg),
        "retrieval": asdict(retrieval_cfg),
        "decode": asdict(decode_cfg),
        "freeform": {
            "method": args.method.lower(),
            "posthoc_extract": args.posthoc_extract,
            "train_output_dir": args.train_output_dir,
        },
        "runtime": {
            "device": "cuda",
            "num_visible_gpus": detect_visible_gpu_count(),
        },
    }
    _write_yaml(output_dir / "resolved_config.yaml", _to_serializable(resolved_config))

    print(
        f"dataset={data_cfg.dataset_name} split={data_cfg.split} method={args.method.lower()} "
        f"posthoc_extract={int(args.posthoc_extract)} device=cuda"
    )
    print(f"config={config_path}")
    print(f"output_dir={output_dir}")

    result = run_bridgeprot_freeform(
        data_config=data_cfg,
        model_config=model_cfg,
        protocol_config=protocol_cfg,
        decode_config=decode_cfg,
        seed=42,
        method_name=args.method.lower(),
        retrieval_config=retrieval_cfg,
        posthoc_extract=args.posthoc_extract,
    )
    _write_result_artifacts(
        output_dir=output_dir / "stage1",
        run_timestamp=run_timestamp,
        result=result,
    )

    write_bridgeprot_task_summary(
        output_dir=output_dir,
        kind="infer-single",
        task_metadata={
            "dataset": data_cfg.dataset_name.lower(),
            "split": data_cfg.split,
            "method": args.method.lower(),
            "protocol": "freeform",
            "posthoc_extract": args.posthoc_extract,
        },
    )

    print(
        f"strict_pair_f1={result.summary.strict_pair_f1:.4f} "
        f"salvaged_pair_f1={result.summary.salvaged_pair_f1:.4f} "
        f"parsability={result.summary.parsability:.4f} "
        f"strict_validity_rate={result.summary.strict_validity_rate:.4f}"
    )


if __name__ == "__main__":
    main()
