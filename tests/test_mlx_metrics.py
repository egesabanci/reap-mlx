"""Tests for MLX pruning metric accumulation."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reap.metrics import PruningState
from reap.router import RouterResult


def test_pruning_state_initialize_shapes_and_dtypes():
    state = PruningState.initialize(3)

    assert state.num_experts == 3
    assert state.total_tokens == 0
    assert state.expert_frequency.shape == (3,)
    assert state.expert_frequency.dtype == np.int64
    assert state.pairwise_expert_frequency is None
    assert state.ean_sum.dtype == np.float64
    assert state.weighted_ean_sum.dtype == np.float64
    assert state.weighted_expert_frequency_sum.dtype == np.float64
    assert state.max_activations.dtype == np.float32

    report = state.report()
    assert report["total_tokens"] == 0
    np.testing.assert_array_equal(report["expert_frequency"], np.zeros(3))
    assert "pairwise_expert_frequency" not in report
    np.testing.assert_allclose(report["expert_proba"], np.zeros(3))
    np.testing.assert_allclose(report["ean_mean"], np.zeros(3))
    np.testing.assert_allclose(report["reap"], np.zeros(3))


def test_metrics_module_import_does_not_import_heavy_runtime_packages():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    code = textwrap.dedent(
        """
        import sys

        BLOCKED_ROOTS = ("torch", "vllm", "mlx", "mlx_lm")

        def is_blocked(fullname):
            return any(
                fullname == root or fullname.startswith(root + ".")
                for root in BLOCKED_ROOTS
            )

        class ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if is_blocked(fullname):
                    raise AssertionError(
                        "forbidden import during MLX metrics import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.metrics import PruningState

        assert PruningState.initialize(2).report()["total_tokens"] == 0

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX metrics import: "
                + ", ".join(forbidden_loaded)
            )
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_accumulate_from_selected_outputs_topk_one():
    state = PruningState.initialize(2)

    indices = np.array([[0], [1], [0]], dtype=np.int64)
    scores = np.array([[0.8], [0.5], [0.2]], dtype=np.float32)
    selected_outputs = np.array(
        [
            [[3.0, 4.0]],
            [[0.0, 6.0]],
            [[5.0, 12.0]],
        ],
        dtype=np.float32,
    )

    state.accumulate(
        indices=indices,
        scores=scores,
        selected_outputs=selected_outputs,
    )

    report = state.report()
    assert report["total_tokens"] == 3
    np.testing.assert_array_equal(report["expert_frequency"], [2, 1])
    np.testing.assert_allclose(report["expert_proba"], [2 / 3, 1 / 3])
    np.testing.assert_allclose(report["ean_sum"], [18.0, 6.0])
    np.testing.assert_allclose(report["ean_mean"], [9.0, 6.0])
    np.testing.assert_allclose(report["weighted_ean_sum"], [6.6, 3.0])
    np.testing.assert_allclose(
        report["weighted_expert_frequency_sum"],
        [1.0, 0.5],
    )
    np.testing.assert_allclose(report["reap"], [3.3, 3.0])
    np.testing.assert_allclose(report["max_activations"], [12.0, 6.0])


def test_accumulate_from_precomputed_stats_topk_greater_than_one():
    state = PruningState.initialize(3)

    indices = np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int64)
    scores = np.array([[0.7, 0.3], [0.9, 0.4], [0.6, 0.2]])
    norms = np.array([[5.0, 2.0], [3.0, 7.0], [13.0, 11.0]])
    maxes = np.array([[4.0, 1.0], [2.0, 6.0], [12.0, 10.0]])

    state.accumulate(
        indices=indices,
        scores=scores,
        selected_output_norms=norms,
        selected_output_maxes=maxes,
    )

    report = state.report()
    assert report["total_tokens"] == 3
    np.testing.assert_array_equal(report["expert_frequency"], [2, 2, 2])
    np.testing.assert_allclose(report["expert_proba"], [2 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(report["ean_sum"], [12.0, 14.0, 15.0])
    np.testing.assert_allclose(report["ean_mean"], [6.0, 7.0, 7.5])
    np.testing.assert_allclose(report["weighted_ean_sum"], [6.3, 4.9, 8.4])
    np.testing.assert_allclose(
        report["weighted_expert_frequency_sum"],
        [1.1, 1.1, 0.9],
    )
    np.testing.assert_allclose(report["reap"], [3.15, 2.45, 4.2])
    np.testing.assert_allclose(report["max_activations"], [6.0, 10.0, 12.0])


def test_accumulate_tracks_pairwise_frequency_when_enabled():
    state = PruningState.initialize(2, track_pairwise=True)

    assert state.pairwise_expert_frequency is not None
    assert state.pairwise_expert_frequency.shape == (2, 2)
    assert state.pairwise_expert_frequency.dtype == np.int64

    state.accumulate(
        indices=np.array([[0]], dtype=np.int64),
        scores=np.array([[1.0]]),
        selected_output_norms=np.array([[5.0]]),
        selected_output_maxes=np.array([[4.0]]),
    )
    state.accumulate(
        indices=np.array([[0], [1]], dtype=np.int64),
        scores=np.array([[0.3], [0.7]]),
        selected_output_norms=np.array([[10.0], [2.0]]),
        selected_output_maxes=np.array([[8.0], [9.0]]),
    )

    report = state.report()
    assert report["total_tokens"] == 3
    np.testing.assert_array_equal(report["expert_frequency"], [2, 1])
    np.testing.assert_array_equal(
        report["pairwise_expert_frequency"],
        [[4, 3], [3, 2]],
    )
    np.testing.assert_allclose(report["ean_sum"], [15.0, 2.0])
    np.testing.assert_allclose(report["ean_mean"], [7.5, 2.0])
    np.testing.assert_allclose(report["weighted_ean_sum"], [8.0, 1.4])
    np.testing.assert_allclose(
        report["weighted_expert_frequency_sum"],
        [1.3, 0.7],
    )
    np.testing.assert_allclose(report["reap"], [4.0, 1.4])
    np.testing.assert_allclose(report["max_activations"], [8.0, 9.0])


def test_accumulate_accepts_router_result():
    state = PruningState.initialize(2)
    routing = RouterResult(
        indices=np.array([[1]], dtype=np.int64),
        scores=np.array([[0.25]], dtype=np.float32),
    )

    state.accumulate(
        routing,
        selected_output_norms=np.array([[8.0]]),
        selected_output_maxes=np.array([[7.0]]),
    )

    report = state.report()
    assert report["total_tokens"] == 1
    np.testing.assert_array_equal(report["expert_frequency"], [0, 1])
    np.testing.assert_allclose(report["weighted_ean_sum"], [0.0, 2.0])
    np.testing.assert_allclose(report["reap"], [0.0, 2.0])


def test_empty_accumulation_is_noop_without_selected_outputs():
    state = PruningState.initialize(2)

    state.accumulate(
        indices=np.empty((0, 2), dtype=np.int64),
        scores=np.empty((0, 2), dtype=np.float32),
    )

    report = state.report()
    assert report["total_tokens"] == 0
    np.testing.assert_array_equal(report["expert_frequency"], [0, 0])
    assert "pairwise_expert_frequency" not in report
    np.testing.assert_allclose(report["ean_sum"], [0.0, 0.0])
    np.testing.assert_allclose(report["weighted_ean_sum"], [0.0, 0.0])
    np.testing.assert_allclose(report["max_activations"], [0.0, 0.0])


def test_never_selected_experts_report_finite_zeros():
    state = PruningState.initialize(3)

    state.accumulate(
        indices=np.array([[0]], dtype=np.int64),
        scores=np.array([[0.5]]),
        selected_output_norms=np.array([[4.0]]),
        selected_output_maxes=np.array([[3.0]]),
    )

    report = state.report()
    assert np.isfinite(report["ean_mean"]).all()
    assert np.isfinite(report["reap"]).all()
    np.testing.assert_allclose(report["ean_mean"], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(report["reap"], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(report["expert_proba"], [1.0, 0.0, 0.0])


def test_accumulate_validates_required_stats_and_shapes():
    state = PruningState.initialize(2)

    with pytest.raises(ValueError, match="selected_outputs or selected_output_norms"):
        state.accumulate(
            indices=np.array([[0]], dtype=np.int64),
            scores=np.array([[1.0]]),
        )

    with pytest.raises(ValueError, match="same shape"):
        state.accumulate(
            indices=np.array([[0]], dtype=np.int64),
            scores=np.array([1.0]),
            selected_output_norms=np.array([[1.0]]),
            selected_output_maxes=np.array([[1.0]]),
        )

    with pytest.raises(ValueError, match="selected expert indices"):
        state.accumulate(
            indices=np.array([[2]], dtype=np.int64),
            scores=np.array([[1.0]]),
            selected_output_norms=np.array([[1.0]]),
            selected_output_maxes=np.array([[1.0]]),
        )
