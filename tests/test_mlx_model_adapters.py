"""Tests for MLX model adapter helpers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from reap.model_adapters import (
    Lfm2MoeModelAdapter,
    MoeLayerConfig,
    Qwen3MoeModelAdapter,
    get_model_layers,
    get_shared_expert,
    infer_model_adapter,
    make_attention_mask,
    update_lfm2_moe_config,
    update_qwen3_moe_config,
)


def test_model_adapters_module_import_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during MLX adapter import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.model_adapters import Lfm2MoeModelAdapter, Qwen3MoeModelAdapter

        assert Lfm2MoeModelAdapter().adapter_name == "lfm2_moe"
        assert Qwen3MoeModelAdapter().adapter_name == "qwen3_moe"

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX adapter import: "
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


class DenseMlp:
    pass


class MoeMlp:
    def __init__(
        self,
        *,
        num_experts=None,
        top_k=None,
        num_experts_per_tok=None,
        norm_topk_prob=None,
        shared_expert=None,
        shared_experts=None,
    ):
        self.switch_mlp = object()
        if num_experts is not None:
            self.num_experts = num_experts
        if top_k is not None:
            self.top_k = top_k
        if num_experts_per_tok is not None:
            self.num_experts_per_tok = num_experts_per_tok
        if norm_topk_prob is not None:
            self.norm_topk_prob = norm_topk_prob
        if shared_expert is not None:
            self.shared_expert = shared_expert
        if shared_experts is not None:
            self.shared_experts = shared_experts


class Lfm2FeedForward(MoeMlp):
    pass


def layer_with_mlp(mlp):
    return SimpleNamespace(mlp=mlp)


def layer_with_feed_forward(feed_forward):
    return SimpleNamespace(feed_forward=feed_forward)


def model_model_layers(layers):
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def model_layers(layers):
    return SimpleNamespace(layers=layers)


def model_model_model_layers(layers):
    return SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )


def test_get_model_layers_supports_common_mlx_lm_layouts():
    layers_a = [object()]
    layers_b = [object(), object()]
    layers_c = [object(), object(), object()]

    assert get_model_layers(model_model_layers(layers_a)) is layers_a
    assert get_model_layers(model_layers(layers_b)) is layers_b
    assert get_model_layers(model_model_model_layers(layers_c)) is layers_c


def test_get_model_layers_raises_for_unknown_layout():
    with pytest.raises(ValueError, match="Could not find model layers"):
        get_model_layers(SimpleNamespace(model=SimpleNamespace()))


def test_qwen3_adapter_identifies_all_moe_layers():
    adapter = Qwen3MoeModelAdapter()
    layers = [
        layer_with_mlp(MoeMlp()),
        layer_with_mlp(MoeMlp()),
        layer_with_mlp(MoeMlp()),
    ]

    assert adapter.layers(model_model_layers(layers)) is layers
    assert adapter.identify_moe_layers(model_model_layers(layers)) == [0, 1, 2]


def test_qwen3_adapter_identifies_partial_moe_layers():
    adapter = Qwen3MoeModelAdapter()
    layers = [
        layer_with_mlp(DenseMlp()),
        layer_with_mlp(MoeMlp()),
        layer_with_mlp(DenseMlp()),
        layer_with_mlp(MoeMlp()),
    ]

    assert adapter.identify_moe_layers(model_layers(layers)) == [1, 3]


def test_qwen3_adapter_returns_no_moe_layers_for_dense_only_model():
    adapter = Qwen3MoeModelAdapter()
    layers = [layer_with_mlp(DenseMlp()), layer_with_mlp(DenseMlp())]

    assert adapter.identify_moe_layers(model_model_layers(layers)) == []


def test_qwen3_adapter_get_moe_and_dense_mlp_behavior():
    adapter = Qwen3MoeModelAdapter()
    dense_mlp = DenseMlp()
    moe_mlp = MoeMlp()
    dense_layer = layer_with_mlp(dense_mlp)
    moe_layer = layer_with_mlp(moe_mlp)

    assert adapter.get_moe(moe_layer) is moe_mlp
    assert adapter.get_dense_mlp(dense_layer) is dense_mlp

    with pytest.raises(ValueError, match="Qwen3-style MoE"):
        adapter.get_moe(dense_layer)
    with pytest.raises(ValueError, match="mlp module"):
        adapter.get_dense_mlp(SimpleNamespace())


def test_qwen3_adapter_layer_config_prefers_live_attributes():
    adapter = Qwen3MoeModelAdapter()
    layer = layer_with_mlp(
        MoeMlp(num_experts=4, top_k=2, norm_topk_prob=True)
    )
    config = {
        "num_experts": 99,
        "num_experts_per_tok": 8,
        "norm_topk_prob": False,
    }

    layer_config = adapter.get_layer_config(layer, config)

    assert layer_config == MoeLayerConfig(
        num_experts=4,
        top_k=2,
        norm_topk_prob=True,
        adapter_name="qwen3_moe",
    )


def test_qwen3_adapter_layer_config_supports_top_k_config_conventions():
    adapter = Qwen3MoeModelAdapter()
    layer = layer_with_mlp(MoeMlp(num_experts=6))

    with_num_experts_per_tok = adapter.get_layer_config(
        layer,
        {"num_experts_per_tok": 3},
    )
    with_top_k = adapter.get_layer_config(layer, {"top_k": 2})

    assert with_num_experts_per_tok.top_k == 3
    assert with_top_k.top_k == 2


def test_qwen3_adapter_layer_config_supports_live_num_experts_per_tok():
    adapter = Qwen3MoeModelAdapter()
    layer = layer_with_mlp(MoeMlp(num_experts=5, num_experts_per_tok=4))

    layer_config = adapter.get_layer_config(layer, {"num_experts_per_tok": 1})

    assert layer_config.top_k == 4


def test_qwen3_adapter_layer_config_requires_expert_count_and_top_k():
    adapter = Qwen3MoeModelAdapter()

    with pytest.raises(ValueError, match="num_experts is required"):
        adapter.get_layer_config(layer_with_mlp(MoeMlp(top_k=1)), {})

    with pytest.raises(ValueError, match="top_k is required"):
        adapter.get_layer_config(layer_with_mlp(MoeMlp(num_experts=2)), {})


def test_lfm2_adapter_identifies_feed_forward_moe_layers_after_dense_layers():
    adapter = Lfm2MoeModelAdapter()
    layers = [
        layer_with_feed_forward(DenseMlp()),
        layer_with_feed_forward(DenseMlp()),
        layer_with_feed_forward(Lfm2FeedForward()),
        layer_with_feed_forward(Lfm2FeedForward()),
    ]

    model = model_model_layers(layers)

    assert adapter.layers(model) is layers
    assert adapter.identify_moe_layers(model) == [2, 3]
    assert adapter.get_dense_mlp(layers[0]) is layers[0].feed_forward
    assert adapter.get_moe(layers[2]) is layers[2].feed_forward


def test_lfm2_adapter_layer_config_includes_expert_bias_flag():
    adapter = Lfm2MoeModelAdapter()
    feed_forward = Lfm2FeedForward(
        num_experts=32,
        top_k=4,
        norm_topk_prob=True,
    )
    feed_forward.use_expert_bias = True
    layer = layer_with_feed_forward(feed_forward)

    layer_config = adapter.get_layer_config(
        layer,
        {
            "num_experts": 99,
            "num_experts_per_tok": 8,
            "norm_topk_prob": False,
            "use_expert_bias": False,
        },
    )

    assert layer_config == MoeLayerConfig(
        num_experts=32,
        top_k=4,
        norm_topk_prob=True,
        adapter_name="lfm2_moe",
        use_expert_bias=True,
    )


def test_infer_model_adapter_selects_lfm2_from_config_and_layout():
    assert infer_model_adapter(config={"model_type": "lfm2_moe"}).adapter_name == (
        "lfm2_moe"
    )
    assert infer_model_adapter(
        config={"architectures": ["Lfm2MoeForCausalLM"]}
    ).adapter_name == "lfm2_moe"

    lfm2_model = model_model_layers(
        [layer_with_feed_forward(DenseMlp()), layer_with_feed_forward(Lfm2FeedForward())]
    )
    qwen_model = model_model_layers([layer_with_mlp(MoeMlp())])

    assert infer_model_adapter(lfm2_model, {}).adapter_name == "lfm2_moe"
    assert infer_model_adapter(qwen_model, {}).adapter_name == "qwen3_moe"


def test_get_shared_expert_handles_plural_and_singular_names():
    plural_shared = object()
    singular_shared = object()

    assert get_shared_expert(MoeMlp(shared_experts=plural_shared)) is plural_shared
    assert get_shared_expert(MoeMlp(shared_expert=singular_shared)) is singular_shared
    assert get_shared_expert(
        MoeMlp(shared_expert=singular_shared, shared_experts=plural_shared)
    ) is plural_shared
    assert get_shared_expert(MoeMlp()) is None


def test_update_qwen3_moe_config_clamps_top_k_without_model_config():
    config = {"num_experts": 8, "num_experts_per_tok": 4}

    result = update_qwen3_moe_config(config, num_experts=2, top_k=4)

    assert result is config
    assert config == {"num_experts": 2, "num_experts_per_tok": 2}


def test_update_qwen3_moe_config_updates_existing_top_k_key_only():
    config = {"num_experts": 8, "num_experts_per_tok": 4, "top_k": 4}

    update_qwen3_moe_config(config, num_experts=3, top_k=2)

    assert config == {"num_experts": 3, "num_experts_per_tok": 2, "top_k": 2}


def test_update_lfm2_moe_config_preserves_expert_bias_flag_and_clamps_top_k():
    config = {
        "model_type": "lfm2_moe",
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "use_expert_bias": True,
    }

    update_lfm2_moe_config(config, num_experts=2, top_k=4)

    assert config == {
        "model_type": "lfm2_moe",
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "use_expert_bias": True,
    }


def test_make_attention_mask_requires_mlx_lm_when_missing():
    if importlib.util.find_spec("mlx_lm") is not None:
        pytest.skip("MLX-LM is installed in this environment.")

    with pytest.raises(ModuleNotFoundError, match="mlx_lm"):
        make_attention_mask(hidden_states=object())


def test_infer_mixtral_adapter_from_block_sparse_moe_layout():
    from types import SimpleNamespace
    from reap.model_adapters import MixtralMoeModelAdapter, infer_model_adapter

    layer = SimpleNamespace(
        block_sparse_moe=SimpleNamespace(
            switch_mlp=object(),
            num_experts=8,
            num_experts_per_tok=2,
        )
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    adapter = infer_model_adapter(model, {"model_type": "mixtral"})
    assert isinstance(adapter, MixtralMoeModelAdapter)
    assert adapter.identify_moe_layers(model) == [0]
    cfg = adapter.get_layer_config(layer, {"num_local_experts": 8, "num_experts_per_tok": 2})
    assert cfg.num_experts == 8
    assert cfg.top_k == 2


def test_infer_qwen2_moe_config_only():
    from reap.model_adapters import Qwen3MoeModelAdapter, infer_model_adapter

    adapter = infer_model_adapter(None, {"model_type": "qwen2_moe"})
    assert isinstance(adapter, Qwen3MoeModelAdapter)
