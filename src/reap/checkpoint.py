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

_CHECKPOINT_VERSION = 1


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

    payload = {
        "version": _CHECKPOINT_VERSION,
        "model_name": model_name,
        "prune_method": prune_method,
        "compression_ratio": compression_ratio,
        "adapter_name": adapter_name,
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
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')!r}; "
            f"expected {_CHECKPOINT_VERSION}."
        )
    keep = payload.get("keep_by_layer")
    if not isinstance(keep, dict) or not keep:
        raise ValueError("Checkpoint has no keep_by_layer mapping.")
    payload["keep_by_layer"] = {
        int(k): [int(x) for x in v] for k, v in keep.items()
    }
    return payload


__all__ = ["write_checkpoint", "load_checkpoint"]
