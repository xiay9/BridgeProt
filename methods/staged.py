from __future__ import annotations

from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtMethodConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    BridgeProtRetrievalConfig,
    bridgeprot_execution_mode,
    resolve_bridgeprot_stage_methods,
)
from core.runner import BridgeProtResult, build_zeroshot_messages, run_bridgeprot_from_messages


def _validate_stage1_method_config(
    *,
    method_name: str,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
) -> None:
    resolved = method_name.lower()
    if resolved not in {"zeroshot", "lora", "sft"}:
        raise ValueError(
            f"Unsupported BridgeProt stage1 method '{method_name}'. Expected one of: zeroshot, lora, sft."
        )

    if resolved == "lora":
        if data_config.use_video and model_config.lora_adapter_path is not None:
            raise ValueError(
                "multimodal-video LoRA expects a merged checkpoint as model_config.model_name. "
                "Do not pass model_config.lora_adapter_path in video mode."
            )
        return

    if resolved == "sft" and model_config.lora_adapter_path is not None:
        raise ValueError("SFT method does not use model_config.lora_adapter_path.")


def run_bridgeprot_stage1_method(
    *,
    method_name: str,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    retrieval_config: BridgeProtRetrievalConfig,
    seed: int,
    llm=None,
    lora_request=None,
    runtime_info=None,
    request_batch_size: int | None = None,
    progress_callback=None,
    dialogues=None,
) -> BridgeProtResult:
    if dialogues is None:
        from core.modalities import load_bridgeprot_multimodal_split

        loaded_split = load_bridgeprot_multimodal_split(
            data_config,
            max_dialogues=data_config.max_dialogues,
        )
        dialogues = loaded_split.dialogues

    resolved_method = method_name.lower()
    if resolved_method in {"zeroshot", "lora", "sft"}:
        _validate_stage1_method_config(
            method_name=resolved_method,
            data_config=data_config,
            model_config=model_config,
        )
        messages = build_zeroshot_messages(
            dialogues=dialogues,
            data_config=data_config,
            protocol_config=protocol_config,
            output_mode=decode_config.output_mode,
        )
    elif resolved_method == "fewshot":
        from methods.fewshot import build_fewshot_messages

        messages = build_fewshot_messages(
            dialogues=dialogues,
            data_config=data_config,
            model_config=model_config,
            protocol_config=protocol_config,
            decode_config=decode_config,
            retrieval_config=retrieval_config,
        )
    else:
        raise ValueError(
            f"Unsupported BridgeProt stage1 method '{method_name}'. "
            "Expected one of: zeroshot, fewshot, lora, sft."
        )

    return run_bridgeprot_from_messages(
        dialogues=dialogues,
        messages=messages,
        model_config=model_config,
        protocol_config=protocol_config,
        decode_config=decode_config,
        seed=seed,
        execution_mode=bridgeprot_execution_mode(use_video=data_config.use_video),
        llm=llm,
        lora_request=lora_request,
        runtime_info=runtime_info,
        request_batch_size=request_batch_size,
        progress_callback=progress_callback,
    )


def run_bridgeprot_stage_pipeline(
    *,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    retrieval_config: BridgeProtRetrievalConfig,
    method_config: BridgeProtMethodConfig,
    stage1_model_config: BridgeProtModelConfig,
    stage2_model_config: BridgeProtModelConfig | None,
    seed: int,
) -> BridgeProtResult:
    stage, stage1_method, stage2_method = resolve_bridgeprot_stage_methods(method_config)
    from methods.stage2 import run_bridgeprot_stage2

    return run_bridgeprot_stage2(
        data_config=data_config,
        stage1_model_config=stage1_model_config,
        stage2_model_config=stage2_model_config or stage1_model_config,
        protocol_config=protocol_config,
        decode_config=decode_config,
        method_config=method_config,
        retrieval_config=retrieval_config,
        seed=seed,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
    )
