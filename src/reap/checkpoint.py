"""Pipeline checkpointing for REAP pruning.

Write/load a JSON checkpoint capturing the prune decision (``keep_by_layer``
and the pre-prune config) so a failed save phase can be retried without
re-running the expensive observe + prune phases.

The module is intentionally import-light: it only needs ``json`` and the
standard library, so importing it never requires MLX, MLX-LM, or other
runtime packages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# v1: keep indices + basic metadata
# v2: adds dataset/seed/layer filters/saliency summaries for auditability
_CHECKPOINT_VERSION = 2
_SUPPORTED_CHECKPOINT_VERSIONS = frozenset({1, 2})


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/native scalars to JSON-serialisable types."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # numpy scalars/arrays carry ``item``/``tolist`` but are not JSON-native.
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def write_checkpoint(
    path: str | Path,
    *,
    keep_by_layer: Mapping[int, Any],
    config_before_prune: Mapping[str, Any] | None,
    model_name: str,
    prune_method: str,
    compression_ratio: float,
    adapter_name: str,
    dataset_name: str | None = None,
    seed: int | None = None,
    max_samples: int | None = None,
    max_seq_length: int | None = None,
    skip_layer_indices: list[int] | None = None,
    prune_layer_indices: list[int] | None = None,
    per_layer_ratios: dict[int, float] | None = None,
    saliency_by_layer: Mapping[int, Any] | None = None,
    observer_token_totals: Mapping[int, int] | None = None,
) -> Path:
    """Write a JSON checkpoint of the prune decision.

    ``keep_by_layer`` maps MoE layer index -> retained expert indices (numpy
    arrays or lists). The indices are stored as plain ints so the checkpoint
    is human-readable and reloadable without numpy.
    """
    serialised_keep: dict[str, list[int]] = {}
    for layer_idx, keep in keep_by_layer.items():
        if hasattr(keep, "tolist"):
            keep_list = keep.tolist()
        else:
            keep_list = list(keep)
        serialised_keep[str(int(layer_idx))] = [int(x) for x in keep_list]

    serialised_saliency: dict[str, list[float]] | None = None
    if saliency_by_layer:
        serialised_saliency = {}
        for layer_idx, scores in saliency_by_layer.items():
            if hasattr(scores, "tolist"):
                values = scores.tolist()
            else:
                values = list(scores)
            serialised_saliency[str(int(layer_idx))] = [float(x) for x in values]

    payload = {
        "version": _CHECKPOINT_VERSION,
        "model_name": model_name,
        "prune_method": prune_method,
        "compression_ratio": compression_ratio,
        "adapter_name": adapter_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "max_samples": max_samples,
        "max_seq_length": max_seq_length,
        "skip_layer_indices": (
            None if skip_layer_indices is None else [int(i) for i in skip_layer_indices]
        ),
        "prune_layer_indices": (
            None
            if prune_layer_indices is None
            else [int(i) for i in prune_layer_indices]
        ),
        "per_layer_ratios": (
            None
            if per_layer_ratios is None
            else {str(int(k)): float(v) for k, v in per_layer_ratios.items()}
        ),
        "observer_token_totals": (
            None
            if observer_token_totals is None
            else {str(int(k)): int(v) for k, v in observer_token_totals.items()}
        ),
        "saliency_by_layer": serialised_saliency,
        "config_before_prune": _jsonable(config_before_prune or {}),
        "keep_by_layer": serialised_keep,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`write_checkpoint`.

    Returns the payload with ``keep_by_layer`` keys converted back to ints.
    Supports checkpoint schema versions 1 and 2.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version not in _SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Unsupported checkpoint version {version!r}; "
            f"supported versions: {sorted(_SUPPORTED_CHECKPOINT_VERSIONS)}."
        )
    keep = payload.get("keep_by_layer")
    if not isinstance(keep, dict) or not keep:
        raise ValueError("Checkpoint has no keep_by_layer mapping.")
    payload["keep_by_layer"] = {
        int(k): [int(x) for x in v] for k, v in keep.items()
    }
    if payload.get("per_layer_ratios"):
        payload["per_layer_ratios"] = {
            int(k): float(v) for k, v in payload["per_layer_ratios"].items()
        }
    if payload.get("observer_token_totals"):
        payload["observer_token_totals"] = {
            int(k): int(v) for k, v in payload["observer_token_totals"].items()
        }
    if payload.get("saliency_by_layer"):
        payload["saliency_by_layer"] = {
            int(k): [float(x) for x in v]
            for k, v in payload["saliency_by_layer"].items()
        }
    return payload


__all__ = ["write_checkpoint", "load_checkpoint"]
