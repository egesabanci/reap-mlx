"""Save/reload validation helpers for pruned MLX models.

This module is intentionally import-light. MLX-LM is imported only when the
default save, load, or generation helpers are executed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reap.backends.mlx.model_adapters import infer_model_adapter


_WEIGHT_PATTERNS = ("*.safetensors", "*.npz")


@dataclass(frozen=True)
class SaveReloadResult:
    """Result of saving, reloading, and validating a pruned MLX model."""

    output_dir: Path
    reloaded_model: Any
    reloaded_tokenizer: Any
    reloaded_config: Mapping[str, Any]
    expected_expert_count: int
    smoke_result: Any = None
    metrics: Mapping[str, Any] | None = None


def save_pruned_model(
    model: Any,
    tokenizer: Any,
    config: Mapping[str, Any],
    output_dir: str | Path,
    original_model_name: str | Path,
    *,
    adapter: Any | None = None,
    expected_expert_count: int | None = None,
    smoke_fn: Callable[[Any, Any, Mapping[str, Any]], Any] | None = None,
    load_fn: Callable[..., Any] | None = None,
    save_fn: Callable[..., Any] | None = None,
) -> SaveReloadResult:
    """Save a pruned model, reload it, and validate the reloaded artifact."""
    adapter = infer_model_adapter(model, config) if adapter is None else adapter
    output_path = _prepare_output_dir(output_dir)
    expected_count = _expected_expert_count(config, expected_expert_count)
    save_fn = _default_save_fn() if save_fn is None else save_fn
    load_fn = _default_load_fn() if load_fn is None else load_fn
    timings: dict[str, float] = {}

    try:
        save_started = time.perf_counter()
        save_fn(
            dst_path=str(output_path),
            src_path_or_repo=str(original_model_name),
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        timings["save_seconds"] = time.perf_counter() - save_started
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save pruned MLX model to {output_path}."
        ) from exc

    _validate_saved_artifacts(output_path)
    artifact_summary = _artifact_summary(output_path)

    reload_started = time.perf_counter()
    reloaded_model, reloaded_tokenizer, reloaded_config = _load_reloaded_model(
        load_fn,
        output_path,
    )
    timings["reload_seconds"] = time.perf_counter() - reload_started

    validation_started = time.perf_counter()
    _validate_reloaded_config(reloaded_config, expected_count)
    _validate_reloaded_model_shapes(
        reloaded_model,
        reloaded_config,
        adapter=adapter,
        expected_expert_count=expected_count,
    )
    timings["reload_validation_seconds"] = time.perf_counter() - validation_started

    smoke_result = None
    smoke_metrics = {
        "enabled": smoke_fn is not None,
        "completed": False,
        "prompt": "What is your name?",
        "max_tokens": 16,
        "elapsed_seconds": None,
        "generated_token_count": None,
        "result_preview": None,
    }
    if smoke_fn is not None:
        smoke_started = time.perf_counter()
        smoke_result = smoke_fn(
            reloaded_model,
            reloaded_tokenizer,
            reloaded_config,
        )
        smoke_elapsed = time.perf_counter() - smoke_started
        timings["smoke_seconds"] = smoke_elapsed
        smoke_metrics.update(
            {
                "completed": True,
                "elapsed_seconds": smoke_elapsed,
                "generated_token_count": _count_generated_tokens(
                    reloaded_tokenizer,
                    smoke_result,
                ),
                "result_preview": _preview(smoke_result),
            }
        )

    return SaveReloadResult(
        output_dir=output_path,
        reloaded_model=reloaded_model,
        reloaded_tokenizer=reloaded_tokenizer,
        reloaded_config=reloaded_config,
        expected_expert_count=expected_count,
        smoke_result=smoke_result,
        metrics={
            "timings": timings,
            "artifacts": artifact_summary,
            "smoke": smoke_metrics,
        },
    )


def generation_smoke(
    model: Any,
    tokenizer: Any,
    config: Mapping[str, Any] | None = None,
    *,
    prompt: str = "What is your name?",
    max_tokens: int = 16,
    generate_fn: Callable[..., Any] | None = None,
) -> Any:
    """Run a short MLX-LM generation smoke test."""
    del config
    generate_fn = _default_generate_fn() if generate_fn is None else generate_fn

    if getattr(tokenizer, "chat_template", None) and hasattr(
        tokenizer,
        "apply_chat_template",
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    return generate_fn(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )


def _prepare_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise OSError(f"Output path exists and is not a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _expected_expert_count(
    config: Mapping[str, Any],
    expected_expert_count: int | None,
) -> int:
    value = expected_expert_count
    if value is None:
        value = config.get("num_experts")
    if value is None:
        raise ValueError(
            "expected_expert_count is required when config['num_experts'] "
            "is missing."
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"expected_expert_count must be positive, got {value}.")
    return value


def _default_save_fn() -> Callable[..., Any]:
    try:
        from mlx_lm import utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "save_pruned_model requires the optional 'mlx_lm' package when "
            "save_fn is not provided."
        ) from exc
    return utils.save


def _default_load_fn() -> Callable[..., Any]:
    try:
        from mlx_lm import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "save_pruned_model requires the optional 'mlx_lm' package when "
            "load_fn is not provided."
        ) from exc
    return load


def _default_generate_fn() -> Callable[..., Any]:
    try:
        from mlx_lm import generate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "generation_smoke requires the optional 'mlx_lm' package when "
            "generate_fn is not provided."
        ) from exc
    return generate


def _validate_saved_artifacts(output_dir: Path) -> None:
    if not (output_dir / "config.json").is_file():
        raise RuntimeError(f"Saved MLX model is missing config.json in {output_dir}.")

    if not any(output_dir.glob(pattern) for pattern in _WEIGHT_PATTERNS):
        raise RuntimeError(
            "Saved MLX model is missing weight artifacts matching "
            f"{_WEIGHT_PATTERNS} in {output_dir}."
        )


def _artifact_summary(output_dir: Path) -> dict[str, Any]:
    files = []
    seen: set[Path] = set()
    for pattern in _WEIGHT_PATTERNS + ("*.json", "*.model", "*.txt"):
        for path in output_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            files.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                }
            )
    files.sort(key=lambda item: item["path"])
    return {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }


def _load_reloaded_model(load_fn: Callable[..., Any], output_dir: Path) -> tuple[
    Any,
    Any,
    Mapping[str, Any],
]:
    try:
        reload_result = load_fn(str(output_dir), return_config=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reload pruned MLX model from {output_dir}."
        ) from exc

    if not isinstance(reload_result, tuple) or len(reload_result) != 3:
        raise ValueError(
            "Expected mlx_lm.load(..., return_config=True) to return "
            "(model, tokenizer, config)."
        )

    reloaded_model, reloaded_tokenizer, reloaded_config = reload_result
    if not isinstance(reloaded_config, Mapping):
        raise ValueError("Reloaded config must be a mapping.")
    return reloaded_model, reloaded_tokenizer, reloaded_config


def _validate_reloaded_config(
    reloaded_config: Mapping[str, Any],
    expected_expert_count: int,
) -> None:
    actual = reloaded_config.get("num_experts")
    if actual is None:
        raise ValueError("Reloaded config is missing 'num_experts'.")
    actual = int(actual)
    if actual != expected_expert_count:
        raise ValueError(
            "Reloaded config expert count mismatch: "
            f"expected {expected_expert_count}, got {actual}."
        )


def _validate_reloaded_model_shapes(
    reloaded_model: Any,
    reloaded_config: Mapping[str, Any],
    *,
    adapter: Any,
    expected_expert_count: int,
) -> None:
    layer_indices = adapter.identify_moe_layers(reloaded_model)
    if not layer_indices:
        raise ValueError("Reloaded model has no adapter-visible MoE layers.")

    layers = adapter.layers(reloaded_model)
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        layer_config = adapter.get_layer_config(layer, reloaded_config)
        if layer_config.num_experts != expected_expert_count:
            raise ValueError(
                f"Reloaded layer {layer_idx} expert count mismatch: expected "
                f"{expected_expert_count}, got {layer_config.num_experts}."
            )
        _validate_switch_moe_shapes(
            adapter.get_moe(layer),
            expected_expert_count=expected_expert_count,
            layer_idx=layer_idx,
            validate_expert_bias=adapter.adapter_name == "lfm2_moe",
        )


def _validate_switch_moe_shapes(
    moe: Any,
    *,
    expected_expert_count: int,
    layer_idx: int,
    validate_expert_bias: bool,
) -> None:
    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        raise ValueError(f"Reloaded layer {layer_idx} does not expose switch_mlp.")

    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(switch_mlp, projection_name, None)
        if projection is None:
            raise ValueError(
                f"Reloaded layer {layer_idx} switch_mlp is missing "
                f"{projection_name}."
            )
        _validate_first_dim(
            getattr(projection, "weight", None),
            expected_expert_count=expected_expert_count,
            name=f"layer {layer_idx} switch_mlp.{projection_name}.weight",
        )

    gate = getattr(moe, "gate", None)
    if gate is None:
        raise ValueError(f"Reloaded layer {layer_idx} does not expose a gate.")
    _validate_first_dim(
        getattr(gate, "weight", None),
        expected_expert_count=expected_expert_count,
        name=f"layer {layer_idx} gate.weight",
    )

    if validate_expert_bias and (
        getattr(moe, "use_expert_bias", False) or hasattr(moe, "expert_bias")
    ):
        _validate_first_dim(
            getattr(moe, "expert_bias", None),
            expected_expert_count=expected_expert_count,
            name=f"layer {layer_idx} expert_bias",
        )


def _validate_first_dim(
    value: Any,
    *,
    expected_expert_count: int,
    name: str,
) -> None:
    if value is None:
        raise ValueError(f"Reloaded {name} is missing.")

    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError(f"Reloaded {name} must expose a non-empty shape.")

    if int(shape[0]) != expected_expert_count:
        raise ValueError(
            f"Reloaded {name} first dimension mismatch: expected "
            f"{expected_expert_count}, got shape {shape}."
        )


def _count_generated_tokens(tokenizer: Any, text: Any) -> int | None:
    if not isinstance(text, str):
        return None

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None

    try:
        token_ids = encode(text, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = encode(text)
        except Exception:
            return None
    except Exception:
        return None

    try:
        return len(token_ids)
    except TypeError:
        return None


def _preview(value: Any, *, max_chars: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


__all__ = [
    "SaveReloadResult",
    "generation_smoke",
    "save_pruned_model",
]
