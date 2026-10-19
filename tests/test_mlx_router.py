"""Tests for MLX router adapters."""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from reap import router as router_module
from reap.router import (
    Lfm2MoeRouter,
    MixtralMoeRouter,
    Qwen3MoeRouter,
    RouterResult,
)

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
        routed_scaling_factor=None,
    ):
        super().__init__(mx, top_k=top_k, norm_topk_prob=norm_topk_prob)
        self.use_expert_bias = use_expert_bias
        if expert_bias is not None:
            self.expert_bias = mx.array(expert_bias)
        # Only the mlx-lm 0.32 block (sigmoid routing) has this attribute.
        if routed_scaling_factor is not None:
            self.routed_scaling_factor = routed_scaling_factor


class TinyPhiMoe(TinyMlp):
    """PhiMoE block as mlx-lm 0.32 builds it: sparsemixer with jitter_eps."""

    def __init__(self, mx, *, top_k, jitter_eps):
        super().__init__(mx, top_k=top_k)
        self.jitter_eps = jitter_eps


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


class RecordingSwitchMlp:
    """Stand-in switch_mlp that records the experts an mlx-lm block selects.

    Each top-k slot returns a unit vector, so the block output's first k
    features are exactly the routing weights the block applied, in slot order.
    """

    def __init__(self, mx):
        self.mx = mx
        self.indices = None

    def __call__(self, x, indices):
        self.indices = indices
        top_k = indices.shape[-1]
        slot_units = self.mx.eye(top_k, x.shape[-1], dtype=x.dtype)
        return self.mx.broadcast_to(slot_units, (*indices.shape, x.shape[-1]))


def _mlx_lm_block_with_tied_gate(mx, model_type):
    """Build an mlx-lm MoE block whose gate logits tie at the top-2 boundary."""
    import importlib

    arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    common = {
        "model_type": model_type,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "num_key_value_heads": 2,
    }
    extra = {
        "qwen3_moe": {
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 8,
            "head_dim": 4,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "max_position_embeddings": 16,
            "norm_topk_prob": False,
        },
        "qwen2_moe": {
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
        },
        "olmoe": {},
    }[model_type]
    args = arch.ModelArgs.from_dict({**common, **extra})
    block_cls = {
        "qwen3_moe": "Qwen3MoeSparseMoeBlock",
        "qwen2_moe": "Qwen2MoeSparseMoeBlock",
        "olmoe": "OlmoeSparseMoeBlock",
    }[model_type]
    block = getattr(arch, block_cls)(args)
    weight = mx.zeros((4, 8))
    weight[:, 0] = mx.array([1.0, 0.5, 0.5, 0.0])
    block.gate.weight = weight
    block.switch_mlp = RecordingSwitchMlp(mx)
    hidden_states = mx.zeros((1, 1, 8))
    hidden_states[..., 0] = 1.0
    return block, hidden_states


@requires_mlx
@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen2_moe", "olmoe"])
def test_qwen_router_breaks_top_k_ties_like_mlx_lm(model_type):
    import mlx.core as mx

    block, hidden_states = _mlx_lm_block_with_tied_gate(mx, model_type)
    mx.eval(block(hidden_states))
    mlx_lm_selected = sorted(block.switch_mlp.indices.reshape(-1).tolist())

    result = Qwen3MoeRouter(block, {"model_type": model_type})(hidden_states)

    assert sorted(result.indices.reshape(-1).tolist()) == mlx_lm_selected


@requires_mlx
@pytest.mark.parametrize(
    ("model_type", "block_cls"),
    [("mixtral", "MixtralSparseMoeBlock"), ("phimoe", "PhiMoESparseMoeBlock")],
)
def test_mixtral_router_matches_mlx_lm_block_routes_and_weights(model_type, block_cls):
    import importlib

    import mlx.core as mx

    arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    args = arch.ModelArgs.from_dict(
        {
            "model_type": model_type,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            # A wide sparsemixer band (mlx-lm 0.32 PhiMoE) mixes several
            # experts into each weight; other routings ignore this key.
            "router_jitter_noise": 0.3,
        }
    )
    block = getattr(arch, block_cls)(args)
    block.switch_mlp = RecordingSwitchMlp(mx)
    hidden_states = mx.random.normal((1, 5, 8), key=mx.random.key(0))

    output = block(hidden_states)
    result = MixtralMoeRouter(block, {"model_type": model_type})(hidden_states)

    assert result.indices.tolist() == block.switch_mlp.indices.tolist()
    assert_allclose(mx, result.scores, output[..., :2])


@requires_mlx
def test_lfm2_router_matches_mlx_lm_block_routes_and_weights():
    """Replays whichever LFM2 routing the installed mlx-lm implements."""
    import mlx.core as mx
    from mlx_lm.models import lfm2_moe

    args = lfm2_moe.ModelArgs.from_dict(
        {
            "model_type": "lfm2_moe",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_experts": 6,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
            "use_expert_bias": True,
            "num_dense_layers": 0,
            "norm_eps": 1e-5,
            "conv_bias": False,
            "conv_L_cache": 3,
            "layer_types": ["full_attention"],
            # mlx-lm 0.31 ignores this key; 0.32 scales the routing weights.
            "routed_scaling_factor": 1.5,
        }
    )
    block = lfm2_moe.Lfm2MoeSparseMoeBlock(args)
    block.gate.weight = block.gate.weight * 8.0
    block.expert_bias = mx.random.uniform(-0.2, 0.2, (6,), key=mx.random.key(1))
    block.switch_mlp = RecordingSwitchMlp(mx)
    hidden_states = mx.random.normal((1, 7, 8), key=mx.random.key(2))

    output = block(hidden_states)
    result = Lfm2MoeRouter(block, {"model_type": "lfm2_moe"})(hidden_states)

    assert result.indices.tolist() == block.switch_mlp.indices.tolist()
    assert_allclose(mx, result.scores, output[..., :2])


@requires_mlx
def test_lfm2_router_uses_sigmoid_routing_when_block_scales_weights():
    import mlx.core as mx

    moe = TinyLfm2Moe(
        mx,
        top_k=2,
        norm_topk_prob=True,
        use_expert_bias=True,
        expert_bias=[0.0, 0.5, 0.0, 0.0],
        routed_scaling_factor=2.0,
    )
    hidden_states = mx.array([[[1.0, 0.0, 0.0]]])

    result = Lfm2MoeRouter(moe)(hidden_states)

    # Logits [1.0, -0.4, 0.05, 0.3]: the bias lifts expert 1 into the top-2
    # (sigmoid 0.40 + 0.5), but its weight is still the unbiased sigmoid.
    assert sorted(result.indices.reshape(-1).tolist()) == [0, 1]
    weights = mx.sigmoid(moe.gate(hidden_states))
    selected = mx.take_along_axis(weights, result.indices, axis=-1)
    expected = selected / (selected.sum(axis=-1, keepdims=True) + 1e-6) * 2.0
    assert_allclose(mx, result.scores, expected)
    # The weights carry no bias, so saliency uses them directly.
    assert result.saliency_scores is None


@requires_mlx
def test_mixtral_router_uses_sparsemixer_when_block_has_jitter_eps():
    import mlx.core as mx

    moe = TinyPhiMoe(mx, top_k=2, jitter_eps=0.01)
    moe.gate = lambda hidden_states: hidden_states
    hidden_states = mx.array([[2.0, 1.99, 0.0, -1.0]])

    result = MixtralMoeRouter(moe)(hidden_states)

    # Expert 1 is within the 2 * jitter_eps band of expert 0, so expert 0 gets
    # softmax([2.0, 1.99])[0]. Expert 0 is masked for the second pass, which
    # leaves expert 1 alone in its band with weight 1. A plain top-2 softmax
    # would split the weight about evenly instead.
    assert result.indices.tolist() == [[0, 1]]
    first = 1.0 / (1.0 + mx.exp(mx.array(-0.01)))
    assert_allclose(mx, result.scores, mx.array([[first.item(), 1.0]]))


def test_router_normalization_epsilons_match_mlx_lm():
    assert router_module._NORM_EPSILON == 1e-20
    assert router_module._SIGMOID_NORM_EPSILON == 1e-6
    assert "_NORM_EPSILON" in inspect.getsource(Qwen3MoeRouter.__call__)

    softmax_source = inspect.getsource(Lfm2MoeRouter._softmax_routes)
    assert "_NORM_EPSILON" in softmax_source
    assert "_SIGMOID_NORM_EPSILON" not in softmax_source
    assert "1e-20" not in softmax_source
    sigmoid_source = inspect.getsource(Lfm2MoeRouter._sigmoid_routes)
    assert "_SIGMOID_NORM_EPSILON" in sigmoid_source
    assert "1e-6" not in sigmoid_source


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
