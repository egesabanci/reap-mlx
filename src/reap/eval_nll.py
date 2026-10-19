"""Optional post-prune quality signal: mean token NLL on calibration sequences.

Import-light: MLX is only imported when evaluation runs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calibration_mean_nll(
    model: Any,
    sequences: Sequence[Any],
    *,
    max_sequences: int = 8,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Compute mean next-token NLL over a few calibration sequences.

    Uses ``model(input_ids)`` logits when available. Returns a JSON-safe dict
    with ``mean_nll``, ``token_count``, and ``sequence_count``. Failures return
    ``status=skipped`` with an error message rather than raising, so the
    pipeline can still complete.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ModuleNotFoundError as exc:
        return {
            "status": "skipped",
            "error": f"mlx not available: {exc}",
            "mean_nll": None,
            "token_count": 0,
            "sequence_count": 0,
        }

    max_sequences = max(int(max_sequences), 1)
    max_seq_length = max(int(max_seq_length), 2)
    total_nll = 0.0
    total_tokens = 0
    used = 0
    last_error: str | None = None

    for sequence in sequences[:max_sequences]:
        input_ids = (
            sequence.get("input_ids") if isinstance(sequence, Mapping) else sequence
        )
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        if ids.size < 2:
            continue
        ids = ids[:max_seq_length]
        tokens = mx.array(ids)[None, :]
        try:
            outputs = model(tokens)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # logits: [1, seq, vocab]; targets are next tokens. Work in float32:
            # NumPy cannot read bfloat16 buffers, and a bf16 log-softmax over a
            # large vocabulary loses most of the NLL precision anyway.
            logits = outputs.astype(mx.float32)
            if logits.ndim != 3 or logits.shape[1] < 2:
                continue
            log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
            targets = tokens[:, 1:]
            # Gather log-prob of true next token.
            flat_lp = log_probs.reshape(-1, log_probs.shape[-1])
            flat_t = targets.reshape(-1)
            # mx.take_along_axis style gather
            nll = -mx.take_along_axis(
                flat_lp, flat_t[:, None], axis=-1
            ).reshape(-1)
            mx.eval(nll)
            values = np.asarray(nll, dtype=np.float64)
            total_nll += float(values.sum())
            total_tokens += int(values.size)
            used += 1
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "calibration NLL failed on a sequence (%s); continuing",
                last_error,
            )
            continue

    if total_tokens < 1:
        reason = (
            f"last error: {last_error}"
            if last_error is not None
            else "no sequence produced [1, seq, vocab] logits with seq >= 2"
        )
        return {
            "status": "skipped",
            "error": f"no tokens evaluated ({reason})",
            "mean_nll": None,
            "token_count": 0,
            "sequence_count": used,
        }
    return {
        "status": "ok",
        "mean_nll": total_nll / total_tokens,
        "perplexity": float(np.exp(total_nll / total_tokens)),
        "token_count": total_tokens,
        "sequence_count": used,
    }


__all__ = ["calibration_mean_nll"]
