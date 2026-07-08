"""Tests for MLX layerwise observation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reap.observer import observe_model


MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

requires_mlx = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX is not installed in this environment.",
)

EXPECTED_PRUNING_KEYS = {
    "total_tokens",
    "total_slots",
    "expert_frequency",
    "expert_proba",
    "ean_sum",
    "ean_mean",
    "weighted_ean_sum",
    "weighted_expert_frequency_sum",
    "reap",
    "max_activations",
}


def test_observer_module_import_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during MLX observer import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.observer import observe_model

        assert observe_model is not None

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX observer import: "
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


class Identity:
    def __call__(self, x):
        return x


class ZeroAttention:
    def __init__(self, mx):
        self.mx = mx
        self.masks = []

    def __call__(self, x, mask, cache=None):
        self.masks.append(mask)
        return self.mx.zeros_like(x)


class TinyEmbed:
    def __init__(self, mx):
        self.mx = mx

    def __call__(self, tokens):
        token_values = tokens.astype(self.mx.float32)
        return self.mx.stack([token_values, token_values + 1.0], axis=-1)


class TinyGate:
    def __init__(self, mx):
        self.mx = mx

    def __call__(self, hidden_states):
        token_values = hidden_states[:, 0]
        return self.mx.stack(
            [
                1.0 - token_values,
                token_values,
                self.mx.zeros_like(token_values) - 2.0,
            ],
            axis=-1,
        )


class TinySwitchMlp:
    def __init__(self, mx):
        self.mx = mx

    def __call__(self, hidden_states, indices):
        expert_values = indices.astype(self.mx.float32)
        return self.mx.stack([expert_values + 1.0, expert_values + 2.0], axis=-1)


class TinySharedExpert:
    def __init__(self, mx, value):
        self.mx = mx
        self.value = value
        self.call_count = 0

    def __call__(self, hidden_states):
        self.call_count += 1
        return self.mx.ones_like(hidden_states) * self.value


class TinyMoeMlp:
    def __init__(self, mx, *, top_k=1, shared_expert=None):
        self.num_experts = 3
        self.top_k = top_k
        self.norm_topk_prob = False
        self.gate = TinyGate(mx)
        self.switch_mlp = TinySwitchMlp(mx)
        if shared_expert is not None:
            self.shared_experts = shared_expert


class TinyDenseMlp:
    def __init__(self, mx):
        self.mx = mx
        self.inputs = []

    def __call__(self, hidden_states):
        self.inputs.append(np.asarray(hidden_states))
        return self.mx.zeros_like(hidden_states)


class TinyLayer:
    def __init__(self, mx, mlp):
        self.input_layernorm = Identity()
        self.post_attention_layernorm = Identity()
        self.self_attn = ZeroAttention(mx)
        self.mlp = mlp


class TinyModel:
    def __init__(self, mx, layers):
        self.model = type("TinyModelBody", (), {})()
        self.model.embed_tokens = TinyEmbed(mx)
        self.model.layers = layers


class TinyLfmOperator:
    def __init__(self, mx, value):
        self.mx = mx
        self.value = value
        self.calls = []

    def __call__(self, x, mask=None, cache=None):
        self.calls.append((mask, cache))
        return self.mx.ones_like(x) * self.value


class TinyLfmGate:
    def __init__(self, mx):
        self.mx = mx

    def __call__(self, hidden_states):
        token_values = hidden_states[..., 0]
        return self.mx.stack(
            [
                1.0 - token_values,
                token_values,
                self.mx.zeros_like(token_values) - 2.0,
            ],
            axis=-1,
        )


class TinyLfmMoeFeedForward:
    def __init__(self, mx, *, top_k=1):
        self.num_experts = 3
        self.top_k = top_k
        self.norm_topk_prob = False
        self.use_expert_bias = False
        self.gate = TinyLfmGate(mx)
        self.switch_mlp = TinySwitchMlp(mx)


class TinyLfmDenseFeedForward:
    def __init__(self, mx):
        self.mx = mx
        self.inputs = []

    def __call__(self, hidden_states):
        self.inputs.append(np.asarray(hidden_states))
        return self.mx.zeros_like(hidden_states)


class TinyLfmLayer:
    def __init__(self, mx, *, is_attention_layer, feed_forward, operator_value=0.0):
        self.operator_norm = Identity()
        self.ffn_norm = Identity()
        self.is_attention_layer = is_attention_layer
        self.feed_forward = feed_forward
        if is_attention_layer:
            self.self_attn = TinyLfmOperator(mx, operator_value)
        else:
            self.conv = TinyLfmOperator(mx, operator_value)


def _softmax_top_score():
    values = np.exp(np.array([1.0, 0.0, -2.0]))
    return values[0] / values.sum()


def _mask_recorder():
    calls = []

    def mask_fn(hidden_states, cache=None):
        calls.append((hidden_states.shape, cache))
        return {"shape": hidden_states.shape}

    return calls, mask_fn


@requires_mlx
def test_observe_model_reports_pruning_keys_and_manual_values():
    import mlx.core as mx

    model = TinyModel(mx, [TinyLayer(mx, TinyMoeMlp(mx, top_k=1))])
    mask_calls, mask_fn = _mask_recorder()

    observer_data = observe_model(
        model,
        [{"input_ids": [0, 1]}],
        {"num_experts": 3, "num_experts_per_tok": 1},
        mask_fn=mask_fn,
    )

    assert set(observer_data) == {0}
    report = observer_data[0]
    assert set(report) == EXPECTED_PRUNING_KEYS
    assert len(mask_calls) == 1

    top_score = _softmax_top_score()
    norm0 = np.sqrt(5.0)
    norm1 = np.sqrt(13.0)

    assert report["total_tokens"] == 2
    np.testing.assert_array_equal(report["expert_frequency"], [1, 1, 0])
    np.testing.assert_allclose(report["expert_proba"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(report["ean_sum"], [norm0, norm1, 0.0])
    np.testing.assert_allclose(report["ean_mean"], [norm0, norm1, 0.0])
    np.testing.assert_allclose(
        report["weighted_ean_sum"],
        [norm0 * top_score, norm1 * top_score, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        report["weighted_expert_frequency_sum"],
        [top_score, top_score, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        report["reap"],
        [norm0 * top_score, norm1 * top_score, 0.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(report["max_activations"], [2.0, 3.0, 0.0])


@requires_mlx
def test_observe_model_counts_topk_routes_separately_from_tokens():
    import mlx.core as mx

    model = TinyModel(mx, [TinyLayer(mx, TinyMoeMlp(mx, top_k=2))])
    _, mask_fn = _mask_recorder()

    observer_data = observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
        mask_fn=mask_fn,
    )

    report = observer_data[0]
    assert report["total_tokens"] == 2
    assert report["expert_frequency"].sum() == 4
    np.testing.assert_array_equal(report["expert_frequency"], [2, 2, 0])


@requires_mlx
def test_observe_model_reports_only_moe_layers_in_partial_model():
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)

    observer_data = observe_model(
        model,
        [[0]],
        {"num_experts": 3, "num_experts_per_tok": 1},
    )

    assert set(observer_data) == {1}


@requires_mlx
def test_observe_model_reuses_default_qwen_mask_once_per_sequence(monkeypatch):
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)
    mask = object()
    mask_calls = []

    def make_mask(hidden_states, cache=None):
        mask_calls.append((hidden_states.shape, cache))
        return mask

    monkeypatch.setattr("reap.observer.make_attention_mask", make_mask)

    observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
    )

    assert mask_calls == [((1, 2, 2), None)]
    assert [layer.self_attn.masks for layer in layers] == [[mask], [mask], [mask]]


@requires_mlx
def test_observe_model_calls_mask_fn_for_sequence_length_greater_than_one():
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
    ]
    model = TinyModel(mx, layers)
    mask_calls, mask_fn = _mask_recorder()

    observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
        mask_fn=mask_fn,
    )

    assert mask_calls == [((1, 2, 2), None), ((1, 2, 2), None)]
    assert layers[0].self_attn.masks[0] == {"shape": (1, 2, 2)}


@requires_mlx
def test_observe_model_calls_eval_fn_after_each_layer():
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)
    eval_calls = []

    observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
        mask_fn=lambda h, cache=None: None,
        eval_fn=lambda h: eval_calls.append(h.shape),
    )

    assert eval_calls == [(1, 2, 2), (1, 2, 2), (1, 2, 2)]


@requires_mlx
def test_observe_model_eval_frequency_flushes_final_partial_group():
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)
    eval_calls = []

    observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
        eval_frequency=2,
        mask_fn=lambda h, cache=None: None,
        eval_fn=lambda h: eval_calls.append(h.shape),
    )

    assert eval_calls == [(1, 2, 2), (1, 2, 2)]


@requires_mlx
def test_observe_model_eval_frequency_does_not_duplicate_exact_final_group():
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyMoeMlp(mx, top_k=1)),
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)
    eval_calls = []

    observe_model(
        model,
        [[0, 1]],
        {"num_experts": 3, "num_experts_per_tok": 1},
        eval_frequency=2,
        mask_fn=lambda h, cache=None: None,
        eval_fn=lambda h: eval_calls.append(h.shape),
    )

    assert eval_calls == [(1, 2, 2), (1, 2, 2)]


@requires_mlx
def test_observe_model_rejects_invalid_eval_frequency():
    import mlx.core as mx

    model = TinyModel(mx, [TinyLayer(mx, TinyMoeMlp(mx, top_k=1))])

    with pytest.raises(ValueError, match="eval_frequency"):
        observe_model(
            model,
            [[0]],
            {"num_experts": 3, "num_experts_per_tok": 1},
            eval_frequency=0,
        )


@requires_mlx
def test_observe_model_adds_shared_expert_to_hidden_flow_only():
    import mlx.core as mx

    shared_expert = TinySharedExpert(mx, value=5.0)
    dense_mlp = TinyDenseMlp(mx)
    model = TinyModel(
        mx,
        [
            TinyLayer(mx, TinyMoeMlp(mx, top_k=1, shared_expert=shared_expert)),
            TinyLayer(mx, dense_mlp),
        ],
    )

    observer_data = observe_model(
        model,
        [[0]],
        {"num_experts": 3, "num_experts_per_tok": 1},
    )

    top_score = _softmax_top_score()
    expected_hidden = np.array([[[0.0, 1.0]]]) + (
        np.array([[[1.0, 2.0]]]) * top_score
    ) + np.array([[[5.0, 5.0]]])

    assert shared_expert.call_count == 1
    np.testing.assert_allclose(dense_mlp.inputs[0], expected_hidden, rtol=1e-6)
    np.testing.assert_allclose(observer_data[0]["ean_sum"], [np.sqrt(5.0), 0.0, 0.0])


@requires_mlx
def test_observe_model_rejects_empty_calibration_sequence():
    import mlx.core as mx

    model = TinyModel(mx, [TinyLayer(mx, TinyMoeMlp(mx, top_k=1))])

    with pytest.raises(ValueError, match="at least one token"):
        observe_model(
            model,
            [[]],
            {"num_experts": 3, "num_experts_per_tok": 1},
        )


@requires_mlx
def test_observe_model_replays_lfm2_conv_attention_dense_and_moe_layers():
    import mlx.core as mx

    dense_feed_forward = TinyLfmDenseFeedForward(mx)
    conv_layer = TinyLfmLayer(
        mx,
        is_attention_layer=False,
        feed_forward=dense_feed_forward,
        operator_value=0.0,
    )
    moe_layer = TinyLfmLayer(
        mx,
        is_attention_layer=True,
        feed_forward=TinyLfmMoeFeedForward(mx, top_k=1),
        operator_value=0.0,
    )
    model = TinyModel(mx, [conv_layer, moe_layer])
    mask_calls = []
    eval_calls = []

    def mask_fn(hidden_states, cache=None, kind=None):
        mask_calls.append((hidden_states.shape, cache, kind))
        return kind

    observer_data = observe_model(
        model,
        [[0, 1]],
        {
            "model_type": "lfm2_moe",
            "num_experts": 3,
            "num_experts_per_tok": 1,
            "norm_topk_prob": False,
            "use_expert_bias": False,
        },
        mask_fn=mask_fn,
        eval_fn=lambda h: eval_calls.append(h.shape),
    )

    assert set(observer_data) == {1}
    assert set(observer_data[1]) == EXPECTED_PRUNING_KEYS
    assert mask_calls == [
        ((1, 2, 2), None, "attention"),
        ((1, 2, 2), None, "ssm"),
    ]
    assert conv_layer.conv.calls == [("ssm", None)]
    assert moe_layer.self_attn.calls == [("attention", None)]
    assert eval_calls == [(1, 2, 2), (1, 2, 2)]
    assert len(dense_feed_forward.inputs) == 1
    assert np.isfinite(observer_data[1]["reap"]).all()


@requires_mlx
def test_observe_model_lfm2_eval_frequency_flushes_final_partial_group():
    import mlx.core as mx

    model = TinyModel(
        mx,
        [
            TinyLfmLayer(
                mx,
                is_attention_layer=False,
                feed_forward=TinyLfmDenseFeedForward(mx),
            ),
            TinyLfmLayer(
                mx,
                is_attention_layer=True,
                feed_forward=TinyLfmMoeFeedForward(mx, top_k=1),
            ),
            TinyLfmLayer(
                mx,
                is_attention_layer=False,
                feed_forward=TinyLfmDenseFeedForward(mx),
            ),
        ],
    )
    eval_calls = []

    observer_data = observe_model(
        model,
        [[0, 1]],
        {
            "model_type": "lfm2_moe",
            "num_experts": 3,
            "num_experts_per_tok": 1,
            "norm_topk_prob": False,
            "use_expert_bias": False,
        },
        eval_frequency=2,
        mask_fn=lambda h, cache=None, kind=None: kind,
        eval_fn=lambda h: eval_calls.append(h.shape),
    )

    assert set(observer_data) == {1}
    assert eval_calls == [(1, 2, 2), (1, 2, 2)]


@requires_mlx
def test_observe_model_raises_for_all_dense_non_moe_model():
    """observe_model must fail fast with a clear error when no MoE layers exist.

    This is the regression guard for #65/#67: an all-dense (non-MoE) model has no
    observable MoE layers, so the adapter cannot be inferred. observe_model must
    raise a ValueError rather than silently returning {} or crashing with an
    opaque AttributeError downstream.
    """
    import mlx.core as mx

    layers = [
        TinyLayer(mx, TinyDenseMlp(mx)),
        TinyLayer(mx, TinyDenseMlp(mx)),
    ]
    model = TinyModel(mx, layers)

    with pytest.raises(ValueError, match="MoE architecture adapter"):
        observe_model(
            model,
            [[0]],
            {"num_experts": 3, "num_experts_per_tok": 1},
        )


class BadShapeSwitchMlp:
    """switch_mlp that returns the wrong hidden dim to exercise shape validation."""

    def __init__(self, mx):
        self.mx = mx

    def __call__(self, hidden_states, indices):
        expert_values = indices.astype(self.mx.float32)
        # Wrong hidden dim (3 instead of the model's 2) -> shape mismatch.
        return self.mx.stack(
            [expert_values + 1.0, expert_values + 2.0, expert_values + 3.0],
            axis=-1,
        )


@requires_mlx
def test_observe_model_raises_when_switch_mlp_returns_wrong_shape():
    import mlx.core as mx

    moe = TinyMoeMlp(mx, top_k=1)
    moe.switch_mlp = BadShapeSwitchMlp(mx)
    model = TinyModel(mx, [TinyLayer(mx, moe)])

    with pytest.raises(ValueError, match="switch_mlp returned shape") as excinfo:
        observe_model(
            model,
            [{"input_ids": [0, 1]}],
            {"num_experts": 3, "num_experts_per_tok": 1},
        )
    # The error should surface both the actual and expected shapes.
    msg = str(excinfo.value)
    assert "(" in msg and ")" in msg
