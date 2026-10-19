"""Tests for the calibration NLL quality signal."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reap.eval_nll import calibration_mean_nll

MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

requires_mlx = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX is not installed in this environment.",
)


def test_eval_nll_module_import_does_not_import_heavy_runtime_packages():
    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys

        BLOCKED_ROOTS = ("torch", "vllm", "mlx", "mlx_lm")

        class ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == root or fullname.startswith(root + ".")
                    for root in BLOCKED_ROOTS
                ):
                    raise AssertionError(f"forbidden import: {fullname}")
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.eval_nll import calibration_mean_nll

        assert calibration_mean_nll is not None
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


class UniformLogitsModel:
    """Returns all-zero logits, so every next token has probability 1/vocab."""

    def __init__(self, mx, vocab_size, dtype):
        self.mx = mx
        self.vocab_size = vocab_size
        self.dtype = dtype

    def __call__(self, tokens):
        return self.mx.zeros(
            (*tokens.shape, self.vocab_size),
            dtype=self.dtype,
        )


@requires_mlx
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_calibration_mean_nll_handles_low_precision_logits(dtype_name):
    import mlx.core as mx

    model = UniformLogitsModel(mx, vocab_size=32, dtype=getattr(mx, dtype_name))
    sequences = [{"input_ids": np.arange(1, 9, dtype=np.int32)}]

    payload = calibration_mean_nll(model, sequences)

    assert payload["status"] == "ok"
    assert payload["token_count"] == 7
    assert payload["sequence_count"] == 1
    assert payload["mean_nll"] == pytest.approx(math.log(32), rel=1e-6)


@requires_mlx
def test_calibration_mean_nll_reports_the_last_error_when_nothing_evaluates():
    def broken_model(tokens):
        raise RuntimeError("forward exploded")

    payload = calibration_mean_nll(
        broken_model,
        [{"input_ids": np.arange(1, 9, dtype=np.int32)}],
    )

    assert payload["status"] == "skipped"
    assert payload["token_count"] == 0
    assert "RuntimeError: forward exploded" in payload["error"]


@requires_mlx
def test_calibration_mean_nll_runs_on_real_bfloat16_model():
    import mlx.core as mx
    from mlx_lm.models import qwen3_moe

    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=64,
        norm_topk_prob=True,
    )
    model = qwen3_moe.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    payload = calibration_mean_nll(
        model,
        [{"input_ids": np.arange(1, 17, dtype=np.int32)}],
    )

    assert payload["status"] == "ok"
    assert payload["token_count"] == 15
    assert math.isfinite(payload["mean_nll"])
