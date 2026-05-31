"""Command-line entrypoint for MLX REAP expert pruning."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from reap.backends.mlx.data import load_calibration_sequences
from reap.backends.mlx.observer import observe_model
from reap.backends.mlx.prune import prune_experts, resolve_prune_method
from reap.backends.mlx.save import generation_smoke, save_pruned_model


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the MLX pruning CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the MLX-only REAP expert pruning pipeline.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-config-name")
    parser.add_argument("--prune-method", default="reap")
    parser.add_argument("--compression-ratio", type=float, default=0.25)
    parser.add_argument(
        "--max-samples",
        "--num-calibration-sequences",
        dest="max_samples",
        type=int,
        default=128,
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip generation smoke after save/reload validation.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    load_model_fn: Callable[[str], tuple[Any, Any, Mapping[str, Any]]] | None = None,
    load_calibration_sequences_fn: Callable[..., list[Any]] | None = None,
    observe_model_fn: Callable[..., dict[int, dict[str, Any]]] | None = None,
    prune_experts_fn: Callable[..., dict[int, Any]] | None = None,
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
    save_pruned_model_fn = (
        save_pruned_model if save_pruned_model_fn is None else save_pruned_model_fn
    )
    smoke_fn = generation_smoke if smoke_fn is None and not args.no_smoke else smoke_fn
    if args.no_smoke:
        smoke_fn = None

    try:
        _emit(print_fn, "load: loading MLX-LM model")
        model, tokenizer, config = load_model_fn(args.model_name)
        if not isinstance(config, Mapping):
            raise ValueError("MLX-LM load must return a mapping config.")

        _emit(print_fn, "calibrate: loading calibration sequences")
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

        _emit(print_fn, "observe: collecting pruning metrics")
        observer_data = observe_model_fn(
            model,
            calibration_sequences,
            config,
        )

        _emit(print_fn, "prune: mutating selected experts")
        prune_experts_fn(
            model,
            config,
            observer_data,
            args.prune_method,
            args.compression_ratio,
        )

        _emit(print_fn, "save: saving pruned model and validating reload")
        result = save_pruned_model_fn(
            model,
            tokenizer,
            config,
            args.output_dir,
            args.model_name,
            smoke_fn=smoke_fn,
        )
        _emit(print_fn, "reload/smoke: validation complete")
        _emit(print_fn, f"done: saved MLX pruned model to {result.output_dir}")
        return 0
    except KeyboardInterrupt:
        _emit(print_fn, "interrupted: MLX pruning stopped")
        return 130


def _validate_args(args: argparse.Namespace) -> None:
    resolve_prune_method(args.prune_method)
    _validate_positive_int(args.max_samples, "max_samples")
    _validate_positive_int(args.max_seq_length, "max_seq_length")

    ratio = float(args.compression_ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"compression_ratio must be in [0, 1), got {ratio}.")


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

    loaded = load(model_name, return_config=True)
    if not isinstance(loaded, tuple) or len(loaded) != 3:
        raise ValueError(
            "Expected mlx_lm.load(..., return_config=True) to return "
            "(model, tokenizer, config)."
        )
    return loaded


def _emit(print_fn: Callable[[str], Any], message: str) -> None:
    print_fn(f"[reap-mlx] {message}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
