from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from importlib.machinery import ModuleSpec
import inspect
import os
from pathlib import Path
import shutil
import sys
import types

import torch

from core.config import (
    BridgeProtDataConfig,
    BridgeProtProtocolConfig,
    _build_bridgeprot_training_dir_parts,
)
from core.model_views import materialize_text_only_model_view, resolve_model_source_dir
from core.tokenizer_runtime import strip_runtime_tokenization_state
from configs.paths import OUTPUT_ROOT
from utils.hf_env import configure_hf_environment

configure_hf_environment()


@dataclass(slots=True)
class BridgeProtTrainingConfig:
    method: str = "sft"
    run_name: str = "qwen35_9b"
    output_dir: str | None = None
    seed: int = 42
    train_split: str = "train"
    eval_split: str = "valid"
    target_mode: str = "minimal"
    max_length: int = 4096
    max_train_dialogues: int | None = None
    max_eval_dialogues: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_total_limit: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    wandb_enabled: bool = False
    wandb_project: str = "BridgeProt"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    load_in_16bit: bool = True
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def resolve_output_dir(
        self,
        dataset_name: str,
        *,
        use_audio: bool = False,
        use_video: bool = False,
        use_audio_summary: bool = False,
    ) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            Path(
                OUTPUT_ROOT,
                "bridgeprot",
                "train",
                dataset_name.lower(),
                *_build_bridgeprot_training_dir_parts(
                    run_name=self.run_name,
                    method_name=self.method,
                    target_mode=self.target_mode,
                    use_audio=use_audio,
                    use_video=use_video,
                    use_audio_summary=use_audio_summary,
                    timestamp=timestamp,
                ),
            )
        )


@dataclass(slots=True)
class BridgeProtTrainingResult:
    output_dir: Path
    global_step: int
    train_metrics: dict[str, object]
    final_eval_metrics: dict[str, object] | None
    best_model_checkpoint: str | None
    best_eval_loss: float | None
    merged_model_dir: str | None = None


def _to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _latest_eval_metrics(log_history: list[dict[str, object]]) -> dict[str, object] | None:
    for item in reversed(log_history):
        if "eval_loss" in item:
            return dict(item)
    return None


def _make_training_arguments(
    *,
    output_dir: Path,
    train_size: int,
    training_config: BridgeProtTrainingConfig,
) -> object:
    from trl import SFTConfig

    signature = inspect.signature(SFTConfig.__init__)
    steps_per_epoch = max(
        1,
        (train_size + training_config.per_device_train_batch_size - 1)
        // training_config.per_device_train_batch_size,
    )
    optimizer_steps_per_epoch = max(
        1,
        (steps_per_epoch + training_config.gradient_accumulation_steps - 1)
        // training_config.gradient_accumulation_steps,
    )
    total_steps = max(1, int(optimizer_steps_per_epoch * training_config.num_train_epochs))
    warmup_steps = int(total_steps * training_config.warmup_ratio)
    kwargs = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "num_train_epochs": training_config.num_train_epochs,
        "warmup_steps": warmup_steps,
        "logging_steps": training_config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": training_config.save_total_limit,
        "bf16": torch.cuda.is_available(),
        "report_to": ["wandb"] if training_config.wandb_enabled else [],
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_pin_memory": True,
        "max_length": training_config.max_length,
        "dataset_text_field": "text",
        "packing": False,
        "dataset_num_proc": 1,
    }
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = False
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return SFTConfig(**filtered_kwargs)


def _should_skip_checkpoint_artifact(name: str) -> bool:
    exact_names = {
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    }
    prefixes = (
        "optimizer",
        "scheduler",
        "rng_state",
        "events.out.tfevents",
    )
    return name in exact_names or any(name.startswith(prefix) for prefix in prefixes)


def _materialize_best_checkpoint_to_output_dir(
    *,
    best_checkpoint: str | None,
    output_dir: Path,
) -> bool:
    if not best_checkpoint:
        return False

    checkpoint_dir = Path(best_checkpoint)
    if not checkpoint_dir.exists():
        return False

    for artifact in checkpoint_dir.iterdir():
        if _should_skip_checkpoint_artifact(artifact.name):
            continue
        destination = output_dir / artifact.name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if artifact.is_dir() and not artifact.is_symlink():
            shutil.copytree(artifact, destination)
        else:
            shutil.copy2(artifact, destination)
    return True


def _save_merged_lora_model(
    *,
    trainer,
    output_dir: Path,
    tokenizer_or_processor,
) -> Path:
    merged_dir = output_dir / "merged_model"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    model_to_merge = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    if hasattr(model_to_merge, "module"):
        model_to_merge = model_to_merge.module
    if not hasattr(model_to_merge, "merge_and_unload"):
        raise RuntimeError("Expected a PEFT LoRA model with merge_and_unload() for BridgeProt LoRA export.")

    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    strip_runtime_tokenization_state(tokenizer_or_processor)
    tokenizer_or_processor.save_pretrained(merged_dir)
    return merged_dir


def _persist_text_only_model_view(
    *,
    source_dir: str | Path,
    output_dir: Path,
) -> Path:
    source_path = Path(source_dir).resolve()
    target_dir = output_dir / "text_model_view"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_path, target_dir, symlinks=True)
    return target_dir


def _disable_wandb_imports() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    stub = types.ModuleType("wandb")
    stub.__spec__ = ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = stub


def _is_world_process_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _distributed_cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def train_bridgeprot_supervised(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    video_num_frames: int = 4,
    video_max_edge: int = 224,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    training_config: BridgeProtTrainingConfig,
) -> BridgeProtTrainingResult:
    if training_config.wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", training_config.wandb_project)
    else:
        _disable_wandb_imports()

    try:
        from unsloth import FastLanguageModel, FastVisionModel
    except ModuleNotFoundError as exc:
        if exc.name not in {"unsloth", "unsloth_zoo"}:
            raise
        raise ImportError(
            "Unsloth is required for BridgeProt training. Install the official stack with "
            "`pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo`."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the Unsloth training stack. This is not a missing-package error. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    from data.sft_dataset import BridgeProtVideoDataCollator, build_supervised_splits

    if training_config.method.lower() not in {"sft", "lora"}:
        raise ValueError("BridgeProt training method must be either 'sft' or 'lora'.")
    if training_config.target_mode.lower() not in {"minimal", "full", "freeform"}:
        raise ValueError("BridgeProt target_mode must be one of: minimal, full, freeform.")

    output_dir = training_config.resolve_output_dir(
        data_config.dataset_name,
        use_audio=data_config.use_audio,
        use_video=data_config.use_video,
        use_audio_summary=data_config.audio_summary_enabled,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    use_multimodal_video = data_config.use_video
    text_only_model_view = None
    if not use_multimodal_video:
        text_model_name, text_only_model_view = materialize_text_only_model_view(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            prefix="bridgeprot_unsloth_text_",
        )
    else:
        text_model_name = None

    try:
        resolved_model_name = str(resolve_model_source_dir(model_name))
        resolved_tokenizer_name = str(
            resolve_model_source_dir(tokenizer_name or model_name)
        )
        if use_multimodal_video:
            from transformers import AutoProcessor, Trainer

            model, _ = FastVisionModel.from_pretrained(
                model_name=resolved_model_name,
                max_seq_length=training_config.max_length,
                dtype=None,
                load_in_4bit=training_config.load_in_4bit,
                load_in_8bit=training_config.load_in_8bit,
                load_in_16bit=training_config.load_in_16bit,
                full_finetuning=training_config.method.lower() == "sft",
                fast_inference=False,
                trust_remote_code=trust_remote_code,
            )
            processor = AutoProcessor.from_pretrained(
                resolved_tokenizer_name,
                trust_remote_code=trust_remote_code,
            )
            tokenizer = processor
            if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            if _is_world_process_zero():
                print("BridgeProt training_mode=multimodal-video model_path=vision-language")

            if training_config.method.lower() == "lora":
                model = FastVisionModel.get_peft_model(
                    model,
                    r=training_config.lora_r,
                    target_modules="all-linear",
                    lora_alpha=training_config.lora_alpha,
                    lora_dropout=training_config.lora_dropout,
                    bias="none",
                    finetune_vision_layers=True,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    use_gradient_checkpointing="unsloth",
                    random_state=training_config.seed,
                    max_seq_length=training_config.max_length,
                    use_rslora=False,
                    loftq_config=None,
                )
        else:
            from trl import SFTTrainer

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=text_model_name,
                max_seq_length=training_config.max_length,
                dtype=None,
                load_in_4bit=training_config.load_in_4bit,
                load_in_8bit=training_config.load_in_8bit,
                load_in_16bit=training_config.load_in_16bit,
                full_finetuning=training_config.method.lower() == "sft",
                fast_inference=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if _is_world_process_zero():
                print("BridgeProt training_mode=text-only model_path=causal-lm")

            if training_config.method.lower() == "lora":
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=training_config.lora_r,
                    target_modules=list(training_config.lora_target_modules),
                    lora_alpha=training_config.lora_alpha,
                    lora_dropout=training_config.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=training_config.seed,
                    max_seq_length=training_config.max_length,
                    use_rslora=False,
                    loftq_config=None,
                )

        splits = build_supervised_splits(
            model_name=model_name,
            tokenizer_name=None,
            trust_remote_code=trust_remote_code,
            data_config=data_config,
            protocol_config=protocol_config,
            train_split=training_config.train_split,
            eval_split=training_config.eval_split,
            max_length=training_config.max_length,
            target_mode=training_config.target_mode,
            max_train_dialogues=training_config.max_train_dialogues,
            max_eval_dialogues=training_config.max_eval_dialogues,
        )

        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        training_args = _make_training_arguments(
            output_dir=output_dir,
            train_size=len(splits.train_dataset),
            training_config=training_config,
        )

        if use_multimodal_video:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=splits.train_dataset,
                eval_dataset=splits.eval_dataset,
                data_collator=BridgeProtVideoDataCollator(
                    processor=processor,
                    max_length=training_config.max_length,
                    num_frames=video_num_frames,
                    max_edge=video_max_edge,
                ),
            )
        else:
            trainer_signature = inspect.signature(SFTTrainer.__init__)
            trainer_kwargs = {
                "model": model,
                "args": training_args,
                "train_dataset": splits.train_dataset,
                "eval_dataset": splits.eval_dataset,
            }
            if "processing_class" in trainer_signature.parameters:
                trainer_kwargs["processing_class"] = tokenizer
            elif "tokenizer" in trainer_signature.parameters:
                trainer_kwargs["tokenizer"] = tokenizer
            trainer = SFTTrainer(**trainer_kwargs)
        train_output = trainer.train()
        is_main_process = (
            trainer.is_world_process_zero()
            if hasattr(trainer, "is_world_process_zero")
            else _is_world_process_zero()
        )
        merged_model_dir: str | None = None
        if is_main_process:
            best_checkpoint_dir = (
                Path(trainer.state.best_model_checkpoint)
                if trainer.state.best_model_checkpoint
                else None
            )
            if use_multimodal_video and training_config.method.lower() == "sft":
                pass
            elif not _materialize_best_checkpoint_to_output_dir(
                best_checkpoint=trainer.state.best_model_checkpoint,
                output_dir=output_dir,
            ):
                trainer.save_model()
            strip_runtime_tokenization_state(tokenizer)
            if use_multimodal_video and best_checkpoint_dir is not None and best_checkpoint_dir.exists():
                tokenizer.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(output_dir)
            if (
                not use_multimodal_video
                and text_only_model_view is not None
            ):
                _persist_text_only_model_view(
                    source_dir=text_only_model_view.name,
                    output_dir=output_dir,
                )
            if training_config.method.lower() == "lora":
                merged_model_dir = str(
                    _save_merged_lora_model(
                        trainer=trainer,
                        output_dir=output_dir,
                        tokenizer_or_processor=processor if use_multimodal_video else tokenizer,
                    )
                )
        train_result = BridgeProtTrainingResult(
            output_dir=output_dir,
            global_step=trainer.state.global_step,
            train_metrics=dict(train_output.metrics),
            final_eval_metrics=_latest_eval_metrics(trainer.state.log_history),
            best_model_checkpoint=trainer.state.best_model_checkpoint,
            best_eval_loss=float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            merged_model_dir=merged_model_dir,
        )

        if is_main_process:
            with open(output_dir / "train_config.json", "w", encoding="utf-8") as handle:
                import json

                json.dump(
                    _to_jsonable(
                        {
                            "model_name": model_name,
                            "data": asdict(data_config),
                            "protocol": asdict(protocol_config),
                            "training": asdict(training_config),
                        }
                    ),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            with open(output_dir / "train_result.json", "w", encoding="utf-8") as handle:
                import json

                json.dump(
                    _to_jsonable(asdict(train_result)),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _distributed_cleanup()
        return train_result
    finally:
        if text_only_model_view is not None:
            text_only_model_view.cleanup()
