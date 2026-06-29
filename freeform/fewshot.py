from __future__ import annotations

from copy import deepcopy

from core.chat import count_chat_tokens
from core.config import (
    BridgeProtDataConfig,
    BridgeProtDecodeConfig,
    BridgeProtModelConfig,
    BridgeProtProtocolConfig,
    BridgeProtRetrievalConfig,
)
from core.serializer import serialize_dialogue
from freeform.bank import build_freeform_demo_bank
from freeform.prompts import FREEFORM_SYSTEM_PROMPT, build_freeform_user_content
from retrieval.selectors import BridgeProtExampleSelector


def build_freeform_fewshot_messages(
    *,
    dialogues,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    protocol_config: BridgeProtProtocolConfig,
    decode_config: BridgeProtDecodeConfig,
    retrieval_config: BridgeProtRetrievalConfig,
) -> list[list[dict[str, object]]]:
    demo_bank = build_freeform_demo_bank(
        data_config=data_config,
        protocol_config=protocol_config,
        source_split=retrieval_config.bank_split,
        max_examples=retrieval_config.max_bank_size,
    )
    selector = BridgeProtExampleSelector(
        demo_bank,
        config=retrieval_config,
    )

    messages_list: list[list[dict[str, object]]] = []
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
        ranked_examples = selector.rank(
            dialogue=dialogue,
            serialized_dialogue=serialized,
        )
        examples = _select_examples_with_budget(
            dialogue=dialogue,
            serialized_dialogue=serialized,
            ranked_examples=ranked_examples,
            data_config=data_config,
            model_config=model_config,
            decode_config=decode_config,
            protocol=protocol_config,
            retrieval_config=retrieval_config,
        )
        messages = _build_freeform_fewshot_messages(
            dialogue=dialogue,
            serialized_dialogue=serialized,
            examples=examples,
            data_config=data_config,
            protocol=protocol_config,
            audio_lines=audio_lines,
        )
        messages_list.append(messages)
    return messages_list


def _select_examples_with_budget(
    *,
    dialogue,
    serialized_dialogue: str,
    ranked_examples,
    data_config: BridgeProtDataConfig,
    model_config: BridgeProtModelConfig,
    decode_config: BridgeProtDecodeConfig,
    protocol: BridgeProtProtocolConfig,
    retrieval_config: BridgeProtRetrievalConfig,
) -> list:
    selected_examples: list = []
    prompt_budget = max(1, model_config.max_model_len - min(decode_config.max_tokens, 512))

    base_messages = _build_freeform_fewshot_messages(
        dialogue=dialogue,
        serialized_dialogue=serialized_dialogue,
        examples=[],
        data_config=data_config,
        protocol=protocol,
    )
    base_prompt_tokens = count_chat_tokens(
        model_name=model_config.model_name,
        tokenizer_name=model_config.tokenizer_name,
        trust_remote_code=model_config.trust_remote_code,
        messages=base_messages,
    )
    if base_prompt_tokens > prompt_budget:
        return []

    for example in ranked_examples:
        if len(selected_examples) >= retrieval_config.num_shots:
            break
        candidate_examples = selected_examples + [example]
        candidate_messages = _build_freeform_fewshot_messages(
            dialogue=dialogue,
            serialized_dialogue=serialized_dialogue,
            examples=candidate_examples,
            data_config=data_config,
            protocol=protocol,
        )
        prompt_tokens = count_chat_tokens(
            model_name=model_config.model_name,
            tokenizer_name=model_config.tokenizer_name,
            trust_remote_code=model_config.trust_remote_code,
            messages=candidate_messages,
        )
        if prompt_tokens <= prompt_budget:
            selected_examples.append(example)
    return selected_examples


def _build_freeform_fewshot_messages(
    *,
    dialogue,
    serialized_dialogue: str,
    examples,
    data_config: BridgeProtDataConfig,
    protocol: BridgeProtProtocolConfig,
    audio_lines: list[str] | None = None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [{"role": "system", "content": FREEFORM_SYSTEM_PROMPT}]
    for example in examples:
        messages.append(
            {
                "role": "user",
                "content": deepcopy(example.user_content),
            }
        )
        messages.append({"role": "assistant", "content": example.target_json})

    messages.append(
        {
            "role": "user",
            "content": build_freeform_user_content(
                dialogue=dialogue,
                serialized_dialogue=serialized_dialogue,
                protocol=protocol,
                use_video=data_config.use_video,
                audio_lines=audio_lines,
            ),
        }
    )
    return messages


def _collect_dialogue_audio_summary_lines(dialogue) -> list[str]:
    lines: list[str] = []
    for utterance in dialogue.utterances:
        audio_summary = utterance.metadata.get("audio_summary")
        if audio_summary:
            lines.append(f"Turn {utterance.turn}: {audio_summary}")
    return lines
