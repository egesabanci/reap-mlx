"""Tests for MLX expert pruning mutation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reap.prune import (
    apply_keep_indices,
    compute_keep_indices,
    prune_experts,
    resolve_prune_method,
    slice_first_dim,
)


MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

requires_mlx = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX is not installed in this environment.",
)


def test_prune_module_import_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during MLX prune import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.prune import prune_experts

        assert prune_experts is not None

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX prune import: "
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


class SliceModule:
    def __init__(self, num_experts: int, offset: int):
        base = np.arange(num_experts * 6, dtype=np.float32).reshape(
            num_experts,
            2,
            3,
        )
        self.weight = base + offset
        self.scales = (
            np.arange(num_experts * 2, dtype=np.float32).reshape(num_experts, 2)
            + offset
            + 100
        )
        self.biases = (
            np.arange(num_experts * 2, dtype=np.float32).reshape(num_experts, 2)
            + offset
            + 200
        )
        self.bias = (
            np.arange(num_experts * 2, dtype=np.float32).reshape(num_experts, 2)
            + offset
            + 300
        )

    def get(self, name: str):
        return getattr(self, name, None)


class TinySwitchMlp:
    def __init__(self, num_experts: int):
        self.gate_proj = SliceModule(num_experts, 0)
        self.up_proj = SliceModule(num_experts, 1000)
        self.down_proj = SliceModule(num_experts, 2000)

    def __call__(self, hidden_states, indices):
        selected_values = self.gate_proj.weight[indices, 0, 0]
        return np.stack([selected_values, selected_values + 1.0], axis=-1)


class TinyGate:
    def __init__(self, num_experts: int):
        self.weight = (
            np.arange(num_experts * 2, dtype=np.float32).reshape(num_experts, 2)
            + 4000
        )
        self.bias = np.arange(num_experts, dtype=np.float32) + 5000
        self.e_score_correction_bias = (
            np.arange(num_experts, dtype=np.float32) + 6000
        )
        self.num_experts = num_experts
        self.n_routed_experts = num_experts
        self.top_k = 3


class TinyLfmGate(TinyGate):
    def __init__(self, num_experts: int):
        super().__init__(num_experts)
        self.scales = np.arange(num_experts, dtype=np.float32).reshape(num_experts, 1)
        self.biases = (
            np.arange(num_experts, dtype=np.float32).reshape(num_experts, 1) + 100
        )


class TinyMoe:
    def __init__(self, num_experts: int = 4, top_k: int = 3):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_experts_per_tok = top_k
        self.norm_topk_prob = False
        self.gate = TinyGate(num_experts)
        self.switch_mlp = TinySwitchMlp(num_experts)

    def __call__(self, hidden_states, indices):
        selected_outputs = self.switch_mlp(hidden_states, indices)
        scores = np.ones(indices.shape, dtype=np.float32) / indices.shape[-1]
        return (selected_outputs * scores[..., None]).sum(axis=-2)


def make_model(num_experts: int = 4, top_k: int = 3):
    moe = TinyMoe(num_experts=num_experts, top_k=top_k)
    return SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]),
    )


class TinyLfmMoe:
    def __init__(self, num_experts: int = 32, top_k: int = 4):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_experts_per_tok = top_k
        self.norm_topk_prob = True
        self.use_expert_bias = True
        self.gate = TinyLfmGate(num_experts)
        self.switch_mlp = TinySwitchMlp(num_experts)
        self.expert_bias = np.arange(num_experts, dtype=np.float32) + 7000


def make_lfm2_model(num_experts: int = 32, top_k: int = 4):
    moe = TinyLfmMoe(num_experts=num_experts, top_k=top_k)
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(feed_forward=object()),
                SimpleNamespace(feed_forward=moe),
            ],
        ),
    )


def qwen_config(num_experts: int = 4, top_k: int = 3):
    return {
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "top_k": top_k,
        "norm_topk_prob": False,
    }


def lfm2_config(num_experts: int = 32, top_k: int = 4):
    return {
        "model_type": "lfm2_moe",
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "norm_topk_prob": True,
        "use_expert_bias": True,
    }


def test_resolve_prune_method_aliases_and_rejects_unknown_method():
    assert resolve_prune_method("frequency") == "expert_frequency"
    assert (
        resolve_prune_method("weighted_frequency_sum")
        == "weighted_expert_frequency_sum"
    )
    assert resolve_prune_method("reap") == "reap"

    with pytest.raises(ValueError, match="Unsupported prune method"):
        resolve_prune_method("ean_ca")


def test_compute_keep_indices_keeps_highest_saliency_in_ascending_order():
    keep = compute_keep_indices(np.array([0.1, 0.9, 0.2, 0.8]), 2)

    np.testing.assert_array_equal(keep, np.array([1, 3]))


def test_prune_experts_slices_qwen_switch_router_metadata_and_config():
    model = make_model(num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    moe = model.model.layers[0].mlp
    keep = np.array([1, 3])

    originals = {
        "gate_proj_weight": moe.switch_mlp.gate_proj.weight.copy(),
        "gate_proj_scales": moe.switch_mlp.gate_proj.scales.copy(),
        "gate_proj_biases": moe.switch_mlp.gate_proj.biases.copy(),
        "gate_proj_bias": moe.switch_mlp.gate_proj.bias.copy(),
        "up_proj_weight": moe.switch_mlp.up_proj.weight.copy(),
        "down_proj_weight": moe.switch_mlp.down_proj.weight.copy(),
        "gate_weight": moe.gate.weight.copy(),
        "gate_bias": moe.gate.bias.copy(),
        "gate_correction": moe.gate.e_score_correction_bias.copy(),
    }

    keep_by_layer = prune_experts(
        model,
        config,
        {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}},
        "reap",
        0.5,
    )

    np.testing.assert_array_equal(keep_by_layer[0], keep)
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.weight,
        originals["gate_proj_weight"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.scales,
        originals["gate_proj_scales"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.biases,
        originals["gate_proj_biases"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.bias,
        originals["gate_proj_bias"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.up_proj.weight,
        originals["up_proj_weight"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.down_proj.weight,
        originals["down_proj_weight"][keep],
    )
    np.testing.assert_array_equal(moe.gate.weight, originals["gate_weight"][keep])
    np.testing.assert_array_equal(moe.gate.bias, originals["gate_bias"][keep])
    np.testing.assert_array_equal(
        moe.gate.e_score_correction_bias,
        originals["gate_correction"][keep],
    )

    assert moe.num_experts == 2
    assert moe.top_k == 2
    assert moe.num_experts_per_tok == 2
    assert moe.gate.num_experts == 2
    assert moe.gate.n_routed_experts == 2
    assert moe.gate.top_k == 2
    assert config["num_experts"] == 2
    assert config["num_experts_per_tok"] == 2
    assert config["top_k"] == 2


def test_prune_experts_accepts_weighted_frequency_alias():
    model = make_model(num_experts=3, top_k=2)
    config = qwen_config(num_experts=3, top_k=2)

    keep_by_layer = prune_experts(
        model,
        config,
        {0: {"weighted_expert_frequency_sum": np.array([0.1, 0.9, 0.8])}},
        "weighted_frequency_sum",
        1 / 3,
    )

    np.testing.assert_array_equal(keep_by_layer[0], np.array([1, 2]))
    assert config["num_experts"] == 2
    assert config["num_experts_per_tok"] == 2


@requires_mlx
def test_slice_first_dim_handles_mlx_arrays():
    import mlx.core as mx

    module = SimpleNamespace(
        weight=mx.array([[0, 1], [2, 3], [4, 5]]),
        scales=mx.array([[10], [11], [12]]),
    )

    slice_first_dim(
        module,
        np.array([0, 2]),
        num_experts=3,
        field_names=("weight", "scales"),
    )
    mx.eval(module.weight, module.scales)

    assert module.weight.tolist() == [[0, 1], [4, 5]]
    assert module.scales.tolist() == [[10], [12]]


def test_pruned_tiny_moe_forward_keeps_output_shape():
    model = make_model(num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    moe = model.model.layers[0].mlp

    prune_experts(
        model,
        config,
        {0: {"expert_frequency": np.array([1, 4, 2, 3])}},
        "frequency",
        0.5,
    )

    hidden_states = np.ones((1, 2, 2), dtype=np.float32)
    indices = np.array([[[0, 1], [1, 0]]], dtype=np.int64)
    output = moe(hidden_states, indices)

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("compression_ratio", [-0.1, 1.0])
def test_prune_experts_rejects_invalid_compression_ratio(compression_ratio):
    model = make_model()
    config = qwen_config()

    with pytest.raises(ValueError, match="compression_ratio"):
        prune_experts(
            model,
            config,
            {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}},
            "reap",
            compression_ratio,
        )


def test_prune_experts_rejects_missing_observer_layer_data():
    model = make_model()
    config = qwen_config()

    with pytest.raises(ValueError, match="Missing observer data for MoE layer 0"):
        prune_experts(model, config, {}, "reap", 0.5)


def test_prune_experts_raises_clear_message_when_no_moe_architecture_detected():
    # No switch_mlp anywhere + an unrecognized model_type -> adapter is None.
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=object())])
    )
    with pytest.raises(ValueError, match="Cannot detect a supported MoE architecture") as excinfo:
        prune_experts(model, {"model_type": "bert"}, {}, "reap", 0.5)
    # The message should point to supported architectures, not print 'None'.
    assert "Qwen3-MoE" in str(excinfo.value) and "LFM2-MoE" in str(excinfo.value)


def test_prune_experts_rejects_wrong_saliency_length():
    model = make_model(num_experts=4)
    config = qwen_config(num_experts=4)

    with pytest.raises(ValueError, match="expected 4"):
        prune_experts(
            model,
            config,
            {0: {"reap": np.array([0.1, 0.9, 0.2])}},
            "reap",
            0.5,
        )


def test_prune_experts_rejects_missing_switch_projection():
    model = make_model()
    config = qwen_config()
    delattr(model.model.layers[0].mlp.switch_mlp, "up_proj")

    with pytest.raises(ValueError, match="missing up_proj"):
        prune_experts(
            model,
            config,
            {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}},
            "reap",
            0.5,
        )


def test_prune_experts_rejects_slice_field_with_wrong_first_dimension():
    model = make_model(num_experts=4)
    config = qwen_config(num_experts=4)
    model.model.layers[0].mlp.switch_mlp.gate_proj.scales = np.zeros(
        (3, 2),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="scales first dimension"):
        prune_experts(
            model,
            config,
            {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}},
            "reap",
            0.5,
        )


def test_prune_experts_slices_lfm2_switch_gate_expert_bias_and_config():
    model = make_lfm2_model(num_experts=32, top_k=4)
    config = lfm2_config(num_experts=32, top_k=4)
    moe = model.model.layers[1].feed_forward
    keep = np.arange(16, 32)

    originals = {
        "gate_proj_weight": moe.switch_mlp.gate_proj.weight.copy(),
        "gate_proj_scales": moe.switch_mlp.gate_proj.scales.copy(),
        "gate_proj_biases": moe.switch_mlp.gate_proj.biases.copy(),
        "gate_weight": moe.gate.weight.copy(),
        "gate_scales": moe.gate.scales.copy(),
        "gate_biases": moe.gate.biases.copy(),
        "expert_bias": moe.expert_bias.copy(),
    }

    keep_by_layer = prune_experts(
        model,
        config,
        {1: {"reap": np.arange(32, dtype=np.float32)}},
        "reap",
        0.5,
    )

    np.testing.assert_array_equal(keep_by_layer[1], keep)
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.weight,
        originals["gate_proj_weight"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.scales,
        originals["gate_proj_scales"][keep],
    )
    np.testing.assert_array_equal(
        moe.switch_mlp.gate_proj.biases,
        originals["gate_proj_biases"][keep],
    )
    np.testing.assert_array_equal(moe.gate.weight, originals["gate_weight"][keep])
    np.testing.assert_array_equal(moe.gate.scales, originals["gate_scales"][keep])
    np.testing.assert_array_equal(moe.gate.biases, originals["gate_biases"][keep])
    np.testing.assert_array_equal(moe.expert_bias, originals["expert_bias"][keep])

    assert moe.num_experts == 16
    assert moe.top_k == 4
    assert moe.num_experts_per_tok == 4
    assert config["num_experts"] == 16
    assert config["num_experts_per_tok"] == 4
    assert config["use_expert_bias"] is True


def test_prune_experts_clamps_lfm2_top_k_below_retained_count():
    model = make_lfm2_model(num_experts=4, top_k=3)
    config = lfm2_config(num_experts=4, top_k=3)
    moe = model.model.layers[1].feed_forward

    prune_experts(
        model,
        config,
        {1: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}},
        "reap",
        0.75,
    )

    assert moe.num_experts == 1
    assert moe.top_k == 1
    assert moe.num_experts_per_tok == 1
    assert config["num_experts"] == 1
    assert config["num_experts_per_tok"] == 1


def make_multi_moe_model(num_layers: int = 2, num_experts: int = 4, top_k: int = 3):
    """Build a Qwen3-style model with several MoE layers for selective pruning tests."""
    layers = [
        SimpleNamespace(mlp=TinyMoe(num_experts=num_experts, top_k=top_k))
        for _ in range(num_layers)
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_prune_experts_prune_layer_indices_selecting_all_layers_works():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.3, 0.7, 0.4, 0.6])},
    }
    keep_by_layer = prune_experts(
        model,
        config,
        observer_data,
        "reap",
        0.5,
        prune_layer_indices=[0, 1],
    )
    assert set(keep_by_layer) == {0, 1}
    assert model.model.layers[0].mlp.num_experts == 2
    assert model.model.layers[1].mlp.num_experts == 2
    assert config["num_experts"] == 2


def test_prune_experts_selective_subset_raises_reload_safety_guard():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.3, 0.7, 0.4, 0.6])},
    }
    with pytest.raises(ValueError, match="differing expert counts"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            prune_layer_indices=[0],
        )


def test_prune_experts_skip_layer_indices_raises_reload_safety_guard():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.3, 0.7, 0.4, 0.6])},
    }
    with pytest.raises(ValueError, match="differing expert counts"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            skip_layer_indices=[1],
        )


def test_prune_experts_rejects_invalid_prune_layer_index():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}}
    with pytest.raises(ValueError, match="non-MoE layer indices"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            prune_layer_indices=[0, 99],
        )


def test_prune_experts_rejects_invalid_skip_layer_index():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}}
    with pytest.raises(ValueError, match="non-MoE layer indices"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            skip_layer_indices=[99],
        )


def _make_heterogeneous_model():
    """Two MoE layers with different expert counts (4 and 8)."""
    layers = [
        SimpleNamespace(mlp=TinyMoe(num_experts=4, top_k=3)),
        SimpleNamespace(mlp=TinyMoe(num_experts=8, top_k=3)),
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_prune_experts_per_layer_ratios_uniform_retained_count_works():
    model = _make_heterogeneous_model()
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])},
    }
    # 4 experts * 0.5 -> retain 2; 8 experts * 0.75 -> retain 2. Uniform 2.
    keep_by_layer = prune_experts(
        model,
        config,
        observer_data,
        "reap",
        0.5,
        per_layer_ratios={0: 0.5, 1: 0.75},
    )
    assert set(keep_by_layer) == {0, 1}
    assert model.model.layers[0].mlp.num_experts == 2
    assert model.model.layers[1].mlp.num_experts == 2
    assert config["num_experts"] == 2


def test_prune_experts_per_layer_ratios_differing_counts_raises():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.3, 0.7, 0.4, 0.6])},
    }
    # 4 * 0.5 -> 2; 4 * 0.25 -> 3. Differing -> reload-unsafe.
    with pytest.raises(ValueError, match="differing expert counts"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            per_layer_ratios={0: 0.5, 1: 0.25},
        )


def test_prune_experts_per_layer_ratios_invalid_ratio_raises():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}}
    with pytest.raises(ValueError, match="compression_ratio must be in"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            per_layer_ratios={0: 1.5},
        )


def test_prune_experts_per_layer_ratios_unknown_index_raises():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])}}
    with pytest.raises(ValueError, match="non-MoE layer indices"):
        prune_experts(
            model,
            config,
            observer_data,
            "reap",
            0.5,
            per_layer_ratios={99: 0.5},
        )


def test_apply_keep_indices_reproduces_prune_result_on_fresh_model():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    observer_data = {
        0: {"reap": np.array([0.1, 0.9, 0.2, 0.8])},
        1: {"reap": np.array([0.3, 0.7, 0.4, 0.6])},
    }
    keep_by_layer = prune_experts(model, config, observer_data, "reap", 0.5)

    # Fresh, unpruned model + original config: re-applying the stored keep
    # indices must reproduce the exact same sliced weights and config.
    fresh = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    fresh_config = qwen_config(num_experts=4, top_k=3)
    reapplied = apply_keep_indices(fresh, fresh_config, keep_by_layer)

    assert set(reapplied) == {0, 1}
    for layer_idx in (0, 1):
        np.testing.assert_array_equal(reapplied[layer_idx], keep_by_layer[layer_idx])
        np.testing.assert_array_equal(
            fresh.model.layers[layer_idx].mlp.switch_mlp.gate_proj.weight,
            model.model.layers[layer_idx].mlp.switch_mlp.gate_proj.weight,
        )
        assert fresh.model.layers[layer_idx].mlp.num_experts == 2
    assert fresh_config["num_experts"] == 2


def test_apply_keep_indices_rejects_non_moe_layer_index():
    model = make_multi_moe_model(num_layers=2, num_experts=4, top_k=3)
    config = qwen_config(num_experts=4, top_k=3)
    # Layer index 5 is out of range for a 2-layer model.
    with pytest.raises(ValueError, match="out of range"):
        apply_keep_indices(model, config, {5: np.array([0, 1])})
