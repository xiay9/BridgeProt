from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.model_views import resolve_model_source_dir
from core.tokenizer_runtime import strip_runtime_tokenization_state
from utils.hf_env import configure_hf_environment

configure_hf_environment()

from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str, tokenizer_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_model_source_dir(tokenizer_name or model_name)),
        trust_remote_code=trust_remote_code,
    )
    strip_runtime_tokenization_state(tokenizer)
    return tokenizer


_MEDIA_PLACEHOLDERS = {
    "video": "[Video]",
    "video_url": "[Video]",
    "image": "[Image]",
    "image_url": "[Image]",
    "audio": "[Audio]",
    "audio_url": "[Audio]",
}


def _normalize_messages_for_text_chat_template(
    messages: list[dict[str, object]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "user")
        normalized.append(
            {
                "role": role,
                "content": _flatten_message_content(item.get("content")),
            }
        )
    return normalized


def _flatten_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for item in content:
        text = _flatten_content_item(item)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _flatten_content_item(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return "" if item is None else str(item)

    item_type = str(item.get("type") or "").strip()
    if item_type == "text":
        text = item.get("text")
        return text if isinstance(text, str) else ""
    if item_type in _MEDIA_PLACEHOLDERS:
        return _MEDIA_PLACEHOLDERS[item_type]
    if item_type:
        return f"[{item_type}]"
    return _stringify_unknown_content_item(item)


def _stringify_unknown_content_item(item: dict[str, Any]) -> str:
    text_like = item.get("text")
    if isinstance(text_like, str):
        return text_like
    return ""


def render_chat_prompt(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    messages: list[dict[str, object]],
) -> str:
    tokenizer = _load_tokenizer(
        model_name,
        tokenizer_name or model_name,
        trust_remote_code,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    normalized_messages = _normalize_messages_for_text_chat_template(messages)
    if chat_template:
        return tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return (
        "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in normalized_messages
        )
        + "\nASSISTANT:"
    )


def count_chat_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
    messages: list[dict[str, object]],
) -> int:
    tokenizer = _load_tokenizer(
        model_name,
        tokenizer_name or model_name,
        trust_remote_code,
    )
    prompt_text = render_chat_prompt(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        trust_remote_code=trust_remote_code,
        messages=messages,
    )
    encoded = tokenizer(prompt_text, add_special_tokens=False)
    return int(len(encoded["input_ids"]))
