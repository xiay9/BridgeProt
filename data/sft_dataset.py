from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from datasets import Dataset
import torch

from core.chat import render_chat_prompt
from core.config import BridgeProtDataConfig, BridgeProtProtocolConfig
from core.modalities import load_bridgeprot_multimodal_split, load_bridgeprot_text_split
from core.prompts import build_bridgeprot_chat_messages, build_bridgeprot_messages
from core.serializer import serialize_dialogue
from core.video_decode import sample_video_frames
from data.targets import render_bridgeprot_target_json
from freeform.prompts import build_freeform_chat_messages, build_freeform_messages
from freeform.targets import render_freeform_target_text


@dataclass(slots=True)
class BridgeProtSupervisedSplits:
    train_dataset: Dataset
    eval_dataset: Dataset


@dataclass(slots=True)
class BridgeProtVideoDataCollator:
    processor: object
    max_length: int
    num_frames: int
    max_edge: int

    def __post_init__(self) -> None:
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None and getattr(video_processor, "fps", None) is not None:
            # Use a fixed frame budget for training instead of the processor default fps-based sampling.
            video_processor.fps = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        materialized_messages: list[list[dict[str, Any]]] = []
        batch_video_metadata: list[list[dict[str, object]]] = []
        has_video_metadata = False
        for example in examples:
            messages, video_metadata = _materialize_processor_messages(
                example["messages"],
                num_frames=self.num_frames,
                max_edge=self.max_edge,
            )
            materialized_messages.append(messages)
            batch_video_metadata.append(video_metadata)
            has_video_metadata = has_video_metadata or bool(video_metadata)

        encoded_batch = self.processor.apply_chat_template(
            materialized_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding=True,
            do_sample_frames=False,
            size={"shortest_edge": self.max_edge, "longest_edge": self.max_edge},
            video_metadata=batch_video_metadata if has_video_metadata else None,
        )

        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        batch["labels"][attention_mask == 0] = -100

        for key in ("pixel_values_videos", "video_grid_thw"):
            value = encoded_batch.get(key)
            if value is not None:
                batch[key] = value
        return batch


def build_supervised_splits(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    train_split: str,
    eval_split: str,
    max_length: int,
    target_mode: str = "minimal",
    max_train_dialogues: int | None = None,
    max_eval_dialogues: int | None = None,
) -> BridgeProtSupervisedSplits:
    del max_length
    build_records = _build_multimodal_records_for_split if data_config.use_video else _build_text_records_for_split
    train_records = build_records(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        trust_remote_code=trust_remote_code,
        data_config=data_config,
        protocol_config=protocol_config,
        split=train_split,
        max_dialogues=max_train_dialogues,
        target_mode=target_mode,
    )
    eval_records = build_records(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        trust_remote_code=trust_remote_code,
        data_config=data_config,
        protocol_config=protocol_config,
        split=eval_split,
        max_dialogues=max_eval_dialogues,
        target_mode=target_mode,
    )
    return BridgeProtSupervisedSplits(
        train_dataset=_to_hf_dataset(train_records),
        eval_dataset=_to_hf_dataset(eval_records),
    )


def _to_hf_dataset(records: list[dict[str, object]]) -> Dataset:
    if not records:
        return Dataset.from_dict({"text": []})
    return Dataset.from_list(records)


def _build_text_records_for_split(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    split: str,
    max_dialogues: int | None,
    target_mode: str,
) -> list[dict[str, str]]:
    loaded_split = load_bridgeprot_text_split(data_config, split=split, max_dialogues=max_dialogues)
    dialogues = loaded_split.dialogues

    records: list[dict[str, str]] = []
    for dialogue in dialogues:
        serialized = serialize_dialogue(
            dialogue,
            include_turn_id=data_config.include_turn_id,
            include_speaker=data_config.include_speaker,
        )
        audio_lines = (
            _collect_dialogue_audio_summary_lines(dialogue)
            if data_config.audio_summary_enabled
            else []
        )
        if target_mode == "freeform":
            messages = build_freeform_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
                audio_lines=audio_lines,
            )
            target_text = render_freeform_target_text(
                dialogue,
                protocol=protocol_config,
            )
        else:
            messages = build_bridgeprot_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
                output_mode=target_mode,
                audio_lines=audio_lines,
            )
            target_text = render_bridgeprot_target_json(
                dialogue,
                protocol=protocol_config,
                target_mode=target_mode,
            )
        prompt_text = render_chat_prompt(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            messages=messages,
        )
        records.append({"text": prompt_text + target_text})
    return records


def _build_multimodal_records_for_split(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    data_config: BridgeProtDataConfig,
    protocol_config: BridgeProtProtocolConfig,
    split: str,
    max_dialogues: int | None,
    target_mode: str,
) -> list[dict[str, object]]:
    del model_name, tokenizer_name, trust_remote_code
    loaded_split = load_bridgeprot_multimodal_split(
        data_config,
        split=split,
        max_dialogues=max_dialogues,
    )
    dialogues = loaded_split.dialogues

    records: list[dict[str, object]] = []
    for dialogue in dialogues:
        serialized = serialize_dialogue(
            dialogue,
            include_turn_id=data_config.include_turn_id,
            include_speaker=data_config.include_speaker,
        )
        audio_lines = (
            _collect_dialogue_audio_summary_lines(dialogue)
            if data_config.audio_summary_enabled
            else []
        )
        if target_mode == "freeform":
            messages = build_freeform_chat_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
                use_video=True,
                audio_lines=audio_lines,
            )
            target_text = render_freeform_target_text(
                dialogue,
                protocol=protocol_config,
            )
        else:
            messages = build_bridgeprot_chat_messages(
                dialogue,
                serialized,
                protocol=protocol_config,
                output_mode=target_mode,
                use_video=True,
                audio_lines=audio_lines,
            )
            target_text = render_bridgeprot_target_json(
                dialogue,
                protocol=protocol_config,
                target_mode=target_mode,
            )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
        records.append({"messages": _convert_messages_to_processor_format(messages)})
    return records


def _convert_messages_to_processor_format(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    converted_messages = deepcopy(messages)
    for message in converted_messages:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            content = message["content"]
        if not isinstance(content, list):
            message["content"] = []
            continue
        converted_content: list[dict[str, object]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "video_url":
                converted_content.append(item)
                continue
            video_payload = item.get("video_url")
            if not isinstance(video_payload, dict):
                continue
            url = video_payload.get("url")
            if not isinstance(url, str) or not url:
                continue
            converted_content.append(
                {
                    "type": "video",
                    "video": str(_local_video_path_from_uri(url)),
                }
            )
        message["content"] = converted_content
    return converted_messages


def _collect_dialogue_audio_summary_lines(dialogue) -> list[str]:
    lines: list[str] = []
    for utterance in dialogue.utterances:
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            lines.append(f"Turn {utterance.turn}: {audio_summary}")
    return lines


def _materialize_processor_messages(
    messages: list[dict[str, Any]],
    *,
    num_frames: int,
    max_edge: int,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    sanitized_messages: list[dict[str, Any]] = []
    video_metadata: list[dict[str, object]] = []
    for message in deepcopy(messages):
        role = message.get("role")
        content = message.get("content")
        sanitized_content: list[dict[str, object]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        sanitized_content.append({"type": "text", "text": text})
                elif item_type == "video":
                    video = item.get("video")
                    if isinstance(video, str):
                        frames, metadata = sample_video_frames(
                            video,
                            num_frames=num_frames,
                            max_edge=max_edge,
                        )
                        sanitized_content.append({"type": "video", "video": frames})
                        video_metadata.append(metadata)
        if isinstance(role, str):
            sanitized_messages.append({"role": role, "content": sanitized_content})
    return sanitized_messages, video_metadata


def _local_video_path_from_uri(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(url)
