"""Tests for MLX router adapters."""

from __future__ import annotations

import inspect
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from reap import router as router_module
from reap.router import Lfm2MoeRouter, Qwen3MoeRouter, RouterResult


MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None


def test_router_contracts_import_without_mlx():
    result = RouterResult(indices="indices", scores="scores")

    assert result.indices == "indices"
    assert result.scores == "scores"
    assert result.logits is None
    assert result.score_mode == "actual"
    assert Qwen3MoeRouter is not None


def test_router_module_import_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during MLX router import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.router import Lfm2MoeRouter, Qwen3MoeRouter, RouterResult

        assert Lfm2MoeRouter is not None
        assert Qwen3MoeRouter is not None
        assert RouterResult(indices=1, scores=2).score_mode == "actual"

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX router import: "
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


requires_mlx = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX is not installed in this environment.",
)


class TinyGate:
    def __init__(self, mx, weight, bias=None):
        self.weight = mx.array(weight)
        self.bias = None if bias is None else mx.array(bias)

    def __call__(self, hidden_states):
        logits = hidden_states @ self.weight
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class TinyMlp:
    def __init__(
        self,
        mx,
        *,
        top_k=None,
        norm_topk_prob=None,
    ):
        self.gate = TinyGate(
            mx,
            weight=[
                [1.0, -0.5, 0.25, 0.0],
                [0.0, 0.75, -0.25, 0.5],
                [-0.5, 0.0, 0.5, 1.0],
            ],
            bias=[0.0, 0.1, -0.2, 0.3],
        )
        if top_k is not None:
            self.top_k = top_k
        if norm_topk_prob is not None:
            self.norm_topk_prob = norm_topk_prob


class TinyLfm2Moe(TinyMlp):
    def __init__(
        self,
        mx,
        *,
        top_k=None,
        norm_topk_prob=None,
        use_expert_bias=False,
        expert_bias=None,
    ):
        super().__init__(mx, top_k=top_k, norm_topk_prob=norm_topk_prob)
        self.use_expert_bias = use_expert_bias
        if expert_bias is not None:
            self.expert_bias = mx.array(expert_bias)


def qwen_reference(mx, mlp, hidden_states, top_k, norm_topk_prob):
    leading_shape = hidden_states.shape[:-1]
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    logits = mlp.gate(flat_hidden_states)
    gates = mx.softmax(logits, axis=-1, precise=True)
    indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, indices, axis=-1)

    if norm_topk_prob:
        scores = scores / (
            scores.sum(axis=-1, keepdims=True) + router_module._NORM_EPSILON
        )

    output_shape = (*leading_shape, top_k)
    return indices.reshape(output_shape), scores.reshape(output_shape)


def lfm2_reference(mx, moe, hidden_states, top_k, norm_topk_prob):
    logits = moe.gate(hidden_states).astype(mx.float32)
    gates = mx.softmax(logits, axis=-1)
    if moe.use_expert_bias:
        gates = gates + moe.expert_bias

    indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, indices, axis=-1)
    if norm_topk_prob:
        scores = scores / (
            mx.sum(scores, axis=-1, keepdims=True) + router_module._NORM_EPSILON
        )
    return indices, scores.astype(hidden_states.dtype)


def assert_same_indices(actual, expected):
    assert actual.tolist() == expected.tolist()


def assert_allclose(mx, actual, expected, *, atol=1e-6):
    diff = mx.max(mx.abs(actual - expected))
    mx.eval(diff)
    assert float(diff) <= atol


def test_router_normalization_epsilon_is_shared():
    assert router_module._NORM_EPSILON == 1e-20
    assert "_NORM_EPSILON" in inspect.getsource(Qwen3MoeRouter.__call__)

    lfm2_call_source = inspect.getsource(Lfm2MoeRouter.__call__)
    assert "_NORM_EPSILON" in lfm2_call_source
    assert "1e-20" not in lfm2_call_source


@requires_mlx
def test_qwen_router_accepts_2d_hidden_states():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=2, norm_topk_prob=False)
    hidden_states = mx.array(
        [
            [0.2, 0.4, 0.6],
            [1.0, -0.5, 0.25],
            [-0.25, 0.5, 1.25],
        ]
    )

    result = Qwen3MoeRouter(mlp, {"num_experts_per_tok": 1})(hidden_states)
    expected_indices, expected_scores = qwen_reference(
        mx,
        mlp,
        hidden_states,
        top_k=2,
        norm_topk_prob=False,
    )

    assert result.indices.shape == (3, 2)
    assert result.scores.shape == (3, 2)
    assert result.logits is None
    assert result.score_mode == "actual"
    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)


@requires_mlx
def test_qwen_router_accepts_3d_hidden_states():
    import mlx.core as mx

    mlp = TinyMlp(mx)
    hidden_states = mx.array(
        [
            [[0.2, 0.4, 0.6], [1.0, -0.5, 0.25]],
            [[-0.25, 0.5, 1.25], [0.0, 1.0, -1.0]],
        ]
    )

    result = Qwen3MoeRouter(
        mlp,
        {"num_experts_per_tok": 3, "norm_topk_prob": False},
    )(hidden_states)
    expected_indices, expected_scores = qwen_reference(
        mx,
        mlp,
        hidden_states,
        top_k=3,
        norm_topk_prob=False,
    )

    assert result.indices.shape == (2, 2, 3)
    assert result.scores.shape == (2, 2, 3)
    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)


@requires_mlx
def test_qwen_router_returns_valid_expert_indices():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=2)
    hidden_states = mx.array([[0.5, 0.25, -0.75], [1.0, 1.0, 1.0]])

    result = Qwen3MoeRouter(mlp)(hidden_states)

    min_index = mx.min(result.indices)
    max_index = mx.max(result.indices)
    mx.eval(min_index, max_index)

    assert int(min_index) >= 0
    assert int(max_index) < 4


@requires_mlx
def test_qwen_router_renormalizes_selected_scores_when_enabled():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=2, norm_topk_prob=True)
    hidden_states = mx.array([[0.5, 0.25, -0.75], [1.0, 1.0, 1.0]])

    result = Qwen3MoeRouter(mlp)(hidden_states)
    selected_score_sums = result.scores.sum(axis=-1)

    assert_allclose(mx, selected_score_sums, mx.ones(selected_score_sums.shape))


@requires_mlx
def test_qwen_router_keeps_full_softmax_scores_when_not_renormalized():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=2, norm_topk_prob=False)
    hidden_states = mx.array([[0.5, 0.25, -0.75], [1.0, 1.0, 1.0]])

    result = Qwen3MoeRouter(mlp)(hidden_states)
    expected_indices, expected_scores = qwen_reference(
        mx,
        mlp,
        hidden_states,
        top_k=2,
        norm_topk_prob=False,
    )

    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)
    max_score_sum = mx.max(result.scores.sum(axis=-1))
    mx.eval(max_score_sum)
    assert float(max_score_sum) < 1.0


@requires_mlx
def test_qwen_router_supports_single_token_and_top_k_one():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=1, norm_topk_prob=True)
    hidden_states = mx.array([[0.5, 0.25, -0.75]])

    result = Qwen3MoeRouter(mlp)(hidden_states)
    expected_indices, expected_scores = qwen_reference(
        mx,
        mlp,
        hidden_states,
        top_k=1,
        norm_topk_prob=True,
    )

    assert result.indices.shape == (1, 1)
    assert result.scores.shape == (1, 1)
    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)
    assert_allclose(mx, result.scores.sum(axis=-1), mx.ones((1,)))


@requires_mlx
def test_qwen_router_prefers_live_top_k_over_config():
    import mlx.core as mx

    mlp = TinyMlp(mx, top_k=1, norm_topk_prob=False)
    hidden_states = mx.array([[0.5, 0.25, -0.75]])

    result = Qwen3MoeRouter(mlp, {"num_experts_per_tok": 3})(hidden_states)

    assert result.indices.shape == (1, 1)


@requires_mlx
def test_lfm2_router_matches_reference_without_expert_bias():
    import mlx.core as mx

    moe = TinyLfm2Moe(mx, top_k=2, norm_topk_prob=False)
    hidden_states = mx.array(
        [
            [[0.2, 0.4, 0.6], [1.0, -0.5, 0.25]],
            [[-0.25, 0.5, 1.25], [0.0, 1.0, -1.0]],
        ]
    )

    result = Lfm2MoeRouter(moe)(hidden_states)
    expected_indices, expected_scores = lfm2_reference(
        mx,
        moe,
        hidden_states,
        top_k=2,
        norm_topk_prob=False,
    )

    assert result.indices.shape == (2, 2, 2)
    assert result.scores.shape == (2, 2, 2)
    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)


@requires_mlx
def test_lfm2_router_applies_expert_bias_before_top_k_selection():
    import mlx.core as mx

    moe = TinyLfm2Moe(
        mx,
        top_k=1,
        norm_topk_prob=False,
        use_expert_bias=True,
        expert_bias=[0.0, 0.55, 0.0, 0.0],
    )
    hidden_states = mx.array([[[1.0, 0.0, 0.0]]])

    result = Lfm2MoeRouter(moe)(hidden_states)
    expected_indices, expected_scores = lfm2_reference(
        mx,
        moe,
        hidden_states,
        top_k=1,
        norm_topk_prob=False,
    )

    assert result.indices.tolist() == [[[1]]]
    assert_same_indices(result.indices, expected_indices)
    assert_allclose(mx, result.scores, expected_scores)


@requires_mlx
def test_lfm2_router_renormalizes_biased_scores_with_epsilon():
    import mlx.core as mx

    moe = TinyLfm2Moe(
        mx,
        top_k=3,
        norm_topk_prob=True,
        use_expert_bias=True,
        expert_bias=[0.0, 0.2, 0.1, 0.0],
    )
    hidden_states = mx.array([[[0.5, 0.25, -0.75], [1.0, 1.0, 1.0]]])

    result = Lfm2MoeRouter(moe)(hidden_states)
    selected_score_sums = result.scores.sum(axis=-1)

    assert result.indices.shape == (1, 2, 3)
    assert_allclose(mx, selected_score_sums, mx.ones(selected_score_sums.shape))


@requires_mlx
def test_lfm2_router_saliency_scores_exclude_expert_bias():
    import mlx.core as mx

    moe = TinyLfm2Moe(
        mx,
        top_k=1,
        norm_topk_prob=False,
        use_expert_bias=True,
        expert_bias=[0.0, 0.55, 0.0, 0.0],
    )
    hidden_states = mx.array([[[1.0, 0.0, 0.0]]])

    result = Lfm2MoeRouter(moe)(hidden_states)
    assert result.saliency_scores is not None

    # saliency_scores must be the pure softmax prob at the selected index,
    # i.e. NOT shifted by expert_bias (which would add 0.55 to expert 1).
    pure = mx.softmax(moe.gate(hidden_states), axis=-1)
    expected = mx.take_along_axis(pure, result.indices, axis=-1)
    assert_allclose(mx, result.saliency_scores, expected)
    # Routing scores still carry the bias, so they differ from saliency here.
    assert not mx.allclose(result.scores, result.saliency_scores).item()
