"""Command-line entrypoint for MLX REAP expert pruning."""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from reap.checkpoint import load_checkpoint, write_checkpoint
from reap.data import load_calibration_sequences
from reap.model_adapters import infer_model_adapter
from reap.observer import observe_model
from reap.prune import apply_keep_indices, prune_experts, resolve_prune_method
from reap.save import generation_smoke, save_pruned_model
from reap.validation_metrics import RunMetrics


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the MLX pruning CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the MLX-only REAP expert pruning pipeline.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Calibration dataset name. Required unless "
            "--resume-from-checkpoint is set (resume skips observation)."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Path to a reap-checkpoint.json from a prior run. Loads the "
            "original model and re-applies the stored keep indices, then "
            "retries only the save/reload phase (skips calibration, observe, "
            "and prune)."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-config-name")
    parser.add_argument("--prune-method", default="reap")
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.25,
        help=(
            "Fraction of MoE experts to prune (0 <= ratio < 1). "
            "All MoE layers prune to the same retained expert count; "
            "models with per-layer heterogeneous expert counts are not yet "
            "supported. A value of 0.0 prunes zero experts (no-op)."
        ),
    )
    parser.add_argument(
        "--skip-layer-indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "MoE layer indices to skip during pruning (default: none). "
            "NOTE: mlx-lm reloads require a uniform expert count across all "
            "MoE layers, so skipping layers that would keep a different count "
            "than the pruned layers is not save/reload-safe and will raise."
        ),
    )
    parser.add_argument(
        "--prune-layer-indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Specific MoE layer indices to prune (default: all). "
            "Indices not present in the adapter's MoE layers raise an error."
        ),
    )
    parser.add_argument(
        "--per-layer-ratios",
        nargs="*",
        default=None,
        help=(
            "Per-layer compression ratios as 'index:ratio' pairs, e.g. "
            "--per-layer-ratios 0:0.5 1:0.25. Layers not listed use "
            "--compression-ratio. NOTE: mlx-lm reloads require a uniform "
            "expert count across all MoE layers, so ratios that produce "
            "differing retained counts across layers are not save/reload-safe "
            "and will raise."
        ),
    )
    parser.add_argument(
        "--max-samples",
        "--num-calibration-sequences",
        dest="max_samples",
        type=int,
        default=128,
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help=(
            "Evaluate the MLX graph every N layers during observation. "
            "Higher values reduce GPU syncs but increase peak memory."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metrics-file",
        default="validation-metrics.json",
        help="Metrics JSON filename or absolute path for validation telemetry.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip generation smoke after save/reload validation.",
    )
    parser.add_argument(
        "--smoke-prompt",
        default="What is your name?",
        help="Prompt for the generation smoke test.",
    )
    parser.add_argument(
        "--smoke-max-tokens",
        type=int,
        default=16,
        help="Maximum tokens for the generation smoke test.",
    )
    return parser


def _parse_per_layer_ratios(values: list[str] | None) -> dict[int, float] | None:
    """Parse 'index:ratio' CLI pairs into a {layer_idx: ratio} mapping."""
    if not values:
        return None
    parsed: dict[int, float] = {}
    for item in values:
        if ":" not in item:
            raise ValueError(
                "--per-layer-ratios entries must be 'index:ratio' pairs, "
                f"got {item!r}."
            )
        index_str, _, ratio_str = item.partition(":")
        try:
            index = int(index_str)
            ratio = float(ratio_str)
        except ValueError as exc:
            raise ValueError(
                "--per-layer-ratios entries must be 'index:ratio' with an "
                f"int index and float ratio, got {item!r}."
            ) from exc
        parsed[index] = ratio
    return parsed


def main(
    argv: Sequence[str] | None = None,
    *,
    load_model_fn: Callable[[str], tuple[Any, Any, Mapping[str, Any]]] | None = None,
    load_calibration_sequences_fn: Callable[..., list[Any]] | None = None,
    observe_model_fn: Callable[..., dict[int, dict[str, Any]]] | None = None,
    prune_experts_fn: Callable[..., dict[int, Any]] | None = None,
    apply_keep_indices_fn: Callable[..., dict[int, Any]] | None = None,
    save_pruned_model_fn: Callable[..., Any] | None = None,
    smoke_fn: Callable[[Any, Any, Mapping[str, Any]], Any] | None = None,
    print_fn: Callable[[str], Any] = print,
) -> int:
    """Run the MLX pruning pipeline and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    load_model_fn = _default_load_model if load_model_fn is None else load_model_fn
    load_calibration_sequences_fn = (
        load_calibration_sequences
        if load_calibration_sequences_fn is None
        else load_calibration_sequences_fn
    )
    observe_model_fn = observe_model if observe_model_fn is None else observe_model_fn
    prune_experts_fn = prune_experts if prune_experts_fn is None else prune_experts_fn
    apply_keep_indices_fn = (
        apply_keep_indices if apply_keep_indices_fn is None else apply_keep_indices_fn
    )
    save_pruned_model_fn = (
        save_pruned_model if save_pruned_model_fn is None else save_pruned_model_fn
    )
    if args.no_smoke:
        smoke_fn = None
    elif smoke_fn is None:
        smoke_prompt = args.smoke_prompt
        smoke_max_tokens = args.smoke_max_tokens

        def smoke_fn(model: Any, tokenizer: Any, config: Mapping[str, Any]) -> Any:
            return generation_smoke(
                model,
                tokenizer,
                config,
                prompt=smoke_prompt,
                max_tokens=smoke_max_tokens,
            )

    metrics = RunMetrics(args.output_dir, args.metrics_file)
    metrics.record_runtime()
    metrics.record_run_config(args)
    metrics.sample_memory("start")
    current_phase = "model_load"

    try:
        _emit(print_fn, "load: loading MLX-LM model")
        with metrics.phase("model_load"):
            model, tokenizer, config = load_model_fn(args.model_name)
        if not isinstance(config, Mapping):
            raise ValueError("MLX-LM load must return a mapping config.")
        adapter = _infer_adapter_safely(model, config)
        metrics.record_model_metadata(model, config, adapter=adapter)
        if adapter is None:
            raise ValueError(
                "Could not determine the MoE architecture adapter for this model. "
                "REAP pruning currently supports Qwen3-MoE and LFM2-MoE architectures. "
                f"Model type: {config.get('model_type', 'unknown')}."
            )
        moe_layer_indices = adapter.identify_moe_layers(model)
        if not moe_layer_indices:
            raise ValueError(
                f"Model has no MoE layers detected by the {adapter.adapter_name} adapter. "
                "REAP pruning requires an MoE model with at least one MoE layer."
            )
        metrics.sample_memory("after_model_load")
        if args.resume_from_checkpoint:
            current_phase = "checkpoint_load"
            _emit(print_fn, f"resume: loading checkpoint {args.resume_from_checkpoint}")
            checkpoint = load_checkpoint(args.resume_from_checkpoint)
            ckpt_model = checkpoint.get("model_name")
            if ckpt_model and ckpt_model != args.model_name:
                logger.warning(
                    "Checkpoint was created for model %r but --model-name is "
                    "%r; keep indices may not apply correctly.",
                    ckpt_model, args.model_name,
                )
            ckpt_adapter = checkpoint.get("adapter_name")
            if ckpt_adapter and ckpt_adapter != adapter.adapter_name:
                logger.warning(
                    "Checkpoint adapter %r differs from inferred adapter %r.",
                    ckpt_adapter, adapter.adapter_name,
                )
            config_before_prune = copy.deepcopy(config)
            current_phase = "prune"
            _emit(print_fn, "resume: re-applying pruned experts from checkpoint")
            with metrics.phase("prune"):
                keep_by_layer = apply_keep_indices_fn(
                    model, config, checkpoint["keep_by_layer"], adapter=adapter,
                )
            for layer_idx in sorted(keep_by_layer):
                _emit(
                    print_fn,
                    f"keep: layer {layer_idx} -> {list(map(int, keep_by_layer[layer_idx]))}",
                )
            metrics.record_pruning(
                keep_by_layer,
                config_before=config_before_prune,
                config_after=config,
                observer_data={},
            )
            metrics.sample_memory("after_prune")
        else:
            current_phase = "calibration"
            _emit(print_fn, "calibrate: loading calibration sequences")
            with metrics.phase("calibration"):
                calibration_sequences = load_calibration_sequences_fn(
                    tokenizer,
                    args.dataset_name,
                    split=args.split,
                    dataset_config_name=args.dataset_config_name,
                    max_samples=args.max_samples,
                    max_seq_length=args.max_seq_length,
                    seed=args.seed,
                )
            if not calibration_sequences:
                raise RuntimeError("No non-empty calibration sequences were loaded.")
            metrics.record_calibration(calibration_sequences)
            metrics.sample_memory("after_calibration")

            current_phase = "observe"
            _emit(print_fn, "observe: collecting pruning metrics")
            with metrics.phase("observe"):
                observer_data = observe_model_fn(
                    model,
                    calibration_sequences,
                    config,
                    eval_frequency=args.eval_frequency,
                    print_fn=print_fn,
                )
            metrics.record_observer(observer_data, args.prune_method)
            if not observer_data:
                raise RuntimeError(
                    "Observer returned no data. The model may have no observable MoE layers "
                    "or the adapter could not identify them."
                )
            metrics.sample_memory("after_observe")

            current_phase = "prune"
            _emit(print_fn, "prune: mutating selected experts")
            config_before_prune = copy.deepcopy(config)
            with metrics.phase("prune"):
                keep_by_layer = prune_experts_fn(
                    model,
                    config,
                    observer_data,
                    args.prune_method,
                    args.compression_ratio,
                    skip_layer_indices=args.skip_layer_indices,
                    prune_layer_indices=args.prune_layer_indices,
                    per_layer_ratios=_parse_per_layer_ratios(args.per_layer_ratios),
                )
            metrics.record_pruning(
                keep_by_layer,
                config_before=config_before_prune,
                config_after=config,
                observer_data=observer_data,
            )
            metrics.sample_memory("after_prune")

            # Persist the prune decision so a failed save can be resumed without
            # re-running the expensive observe + prune phases.
            checkpoint_path = Path(args.output_dir) / "reap-checkpoint.json"
            try:
                write_checkpoint(
                    checkpoint_path,
                    keep_by_layer=keep_by_layer,
                    config_before_prune=config_before_prune,
                    model_name=args.model_name,
                    prune_method=args.prune_method,
                    compression_ratio=args.compression_ratio,
                    adapter_name=adapter.adapter_name,
                )
                _emit(print_fn, f"checkpoint: wrote prune decision to {checkpoint_path}")
            except (OSError, ValueError) as exc:
                logger.warning("Failed to write prune checkpoint: %s", exc)
            for layer_idx in sorted(keep_by_layer):
                _emit(
                    print_fn,
                    f"keep: layer {layer_idx} -> {list(map(int, keep_by_layer[layer_idx]))}",
                )

        current_phase = "save_reload_smoke"
        _emit(print_fn, "save: saving pruned model and validating reload")
        with metrics.phase("save_reload_smoke"):
            result = save_pruned_model_fn(
                model,
                tokenizer,
                config,
                args.output_dir,
                args.model_name,
                adapter=adapter,
                smoke_fn=smoke_fn,
                smoke_prompt=args.smoke_prompt,
                smoke_max_tokens=args.smoke_max_tokens,
            )
        metrics.record_save_reload(result, adapter=adapter)
        metrics.sample_memory("after_save_reload_smoke")
        metrics.write(status="success")
        _emit(print_fn, "reload/smoke: validation complete")
        _emit(print_fn, f"done: saved MLX pruned model to {result.output_dir}")
        return 0
    except KeyboardInterrupt:
        _emit(print_fn, "interrupted: MLX pruning stopped")
        return 130
    except Exception as exc:
        try:
            metrics.sample_memory("failure")
            metrics.write(
                status="failed",
                failure=metrics.failure_payload(current_phase, exc),
            )
        except Exception:
            logger.exception("failed to write MLX validation metrics")
        raise


def _validate_args(args: argparse.Namespace) -> None:
    resolve_prune_method(args.prune_method)
    if not args.resume_from_checkpoint and not args.dataset_name:
        raise ValueError(
            "--dataset-name is required unless --resume-from-checkpoint is set "
            "(resume skips calibration and observation)."
        )
    if args.resume_from_checkpoint and not Path(
        args.resume_from_checkpoint
    ).exists():
        raise ValueError(
            f"--resume-from-checkpoint file not found: {args.resume_from_checkpoint}."
        )
    _validate_positive_int(args.max_samples, "max_samples")
    _validate_positive_int(args.max_seq_length, "max_seq_length")
    _validate_positive_int(args.eval_frequency, "eval_frequency")
    _validate_positive_int(args.smoke_max_tokens, "smoke_max_tokens")

    ratio = float(args.compression_ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"compression_ratio must be in [0, 1), got {ratio}.")

    if args.compression_ratio == 0.0:
        logger.warning(
            "compression_ratio=0.0 will prune zero experts; "
            "the saved model will be identical to the input."
        )

    # Parse and validate per-layer ratios early so malformed input surfaces
    # via argparse error handling rather than mid-pipeline.
    _parse_per_layer_ratios(args.per_layer_ratios)


def _validate_positive_int(value: Any, name: str) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def _default_load_model(model_name: str) -> tuple[Any, Any, Mapping[str, Any]]:
    try:
        from mlx_lm import load
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLX entrypoint requires the optional 'mlx_lm' package to load models."
        ) from exc

    if _supports_keyword(load, "return_config"):
        loaded = load(model_name, return_config=True)
        if not isinstance(loaded, tuple) or len(loaded) != 3:
            raise ValueError(
                "Expected mlx_lm.load(..., return_config=True) to return "
                "(model, tokenizer, config)."
            )
        return loaded

    loaded = load(model_name)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError(
            "Expected mlx_lm.load(...) to return (model, tokenizer)."
        )
    model, tokenizer = loaded
    return model, tokenizer, _load_mlx_lm_config(model_name)


def _load_mlx_lm_config(model_name: str) -> Mapping[str, Any]:
    try:
        from mlx_lm.utils import get_model_path, load_config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLX entrypoint requires mlx_lm.utils to load model config."
        ) from exc

    model_path, _ = get_model_path(model_name)
    config = load_config(model_path)
    if not isinstance(config, Mapping):
        raise ValueError("MLX-LM load_config must return a mapping config.")
    return config


def _supports_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _infer_adapter_safely(model: Any, config: Mapping[str, Any]) -> Any | None:
    try:
        return infer_model_adapter(model, config)
    except Exception:
        logger.debug("could not infer MLX model adapter for metrics", exc_info=True)
        return None


def _emit(print_fn: Callable[[str], Any], message: str) -> None:
    print_fn(f"[reap-mlx] {message}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
