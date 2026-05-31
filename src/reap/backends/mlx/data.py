"""Minimal calibration loading for the MLX pruning pipeline.

This module stays independent of the existing Torch/vLLM data pipeline. The
dataset dependency is imported lazily only when calibration data is loaded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def load_calibration_sequences(
    tokenizer: Any,
    dataset_name: str,
    *,
    split: str = "train",
    dataset_config_name: str | None = None,
    max_samples: int = 128,
    max_seq_length: int = 2048,
    seed: int = 42,
    load_dataset_fn: Callable[..., Iterable[Any]] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load unpadded batch-size-1 calibration token sequences."""
    max_samples = _positive_int(max_samples, "max_samples")
    max_seq_length = _positive_int(max_seq_length, "max_seq_length")
    load_dataset_fn = _default_load_dataset if load_dataset_fn is None else load_dataset_fn

    load_kwargs: dict[str, Any] = {"split": split}
    if dataset_config_name is not None:
        load_kwargs["name"] = dataset_config_name
    dataset = load_dataset_fn(dataset_name, **load_kwargs)
    dataset = _maybe_shuffle(dataset, seed)

    sequences: list[dict[str, np.ndarray]] = []
    for record in dataset:
        text = extract_text(record, tokenizer=tokenizer)
        token_ids = tokenize_text(
            tokenizer,
            text,
            max_seq_length=max_seq_length,
        )
        if not token_ids:
            continue

        sequences.append(
            {
                "input_ids": np.asarray(token_ids, dtype=np.int32),
            }
        )
        if len(sequences) >= max_samples:
            break

    return sequences


def extract_text(record: Any, *, tokenizer: Any | None = None) -> str:
    """Extract text from common calibration dataset record shapes."""
    if not isinstance(record, Mapping):
        return _normalize_content(record).strip()

    for field_name in ("messages", "conversations"):
        messages = _maybe_json_load(record.get(field_name))
        if messages:
            return _messages_to_text(messages, tokenizer=tokenizer).strip()

    text = record.get("text")
    if text is not None:
        return _normalize_content(text).strip()

    instruction = _first_present(record, "instruction", "prompt", "question")
    output = _first_present(record, "output", "completion", "response", "answer")
    input_text = _first_present(record, "input", "context")
    if instruction is not None or output is not None:
        parts = [
            _normalize_content(part).strip()
            for part in (instruction, input_text, output)
            if part is not None
        ]
        return "\n\n".join(part for part in parts if part)

    return ""


def tokenize_text(
    tokenizer: Any,
    text: str,
    *,
    max_seq_length: int,
) -> list[int]:
    """Tokenize text and truncate to ``max_seq_length`` tokens."""
    max_seq_length = _positive_int(max_seq_length, "max_seq_length")
    text = text.strip()
    if not text:
        return []

    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = encode(text, add_special_tokens=True)
        except TypeError:
            token_ids = encode(text)
    elif callable(tokenizer):
        encoded = tokenizer(text)
        token_ids = _input_ids_from_encoded(encoded)
    else:
        raise TypeError("tokenizer must expose encode(...) or be callable.")

    token_ids = _flatten_token_ids(token_ids)
    return [int(token_id) for token_id in token_ids[:max_seq_length]]


def _default_load_dataset(*args: Any, **kwargs: Any) -> Iterable[Any]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_calibration_sequences requires the optional 'datasets' package "
            "when load_dataset_fn is not provided."
        ) from exc
    return load_dataset(*args, **kwargs)


def _positive_int(value: Any, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _maybe_shuffle(dataset: Iterable[Any], seed: int) -> Iterable[Any]:
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        return shuffle(seed=seed)

    return dataset


def _maybe_json_load(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _messages_to_text(messages: Any, *, tokenizer: Any | None) -> str:
    if not isinstance(messages, list):
        return _normalize_content(messages)

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            rendered = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            rendered = apply_chat_template(messages, tokenize=False)
        return _normalize_content(rendered)

    parts = []
    for message in messages:
        if not isinstance(message, Mapping):
            content = _normalize_content(message).strip()
            if content:
                parts.append(content)
            continue

        role = _first_present(message, "role", "from", "speaker")
        content = _first_present(message, "content", "value", "text")
        content_text = _normalize_content(content).strip()
        if not content_text:
            continue
        role_text = _normalize_content(role).strip()
        if role_text:
            parts.append(f"{role_text}: {content_text}")
        else:
            parts.append(content_text)
    return "\n".join(parts)


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(dict(item), sort_keys=True))
            else:
                parts.append(_normalize_content(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        return json.dumps(dict(content), sort_keys=True)
    return str(content)


def _first_present(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _input_ids_from_encoded(encoded: Any) -> Any:
    if isinstance(encoded, Mapping):
        return encoded["input_ids"]
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is not None:
        return input_ids
    raise TypeError("Callable tokenizer output must expose input_ids.")


def _flatten_token_ids(token_ids: Any) -> list[Any]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if (
        isinstance(token_ids, Sequence)
        and token_ids
        and isinstance(token_ids[0], Sequence)
        and not isinstance(token_ids[0], (str, bytes))
    ):
        token_ids = token_ids[0]
    return list(token_ids)


__all__ = [
    "extract_text",
    "load_calibration_sequences",
    "tokenize_text",
]
