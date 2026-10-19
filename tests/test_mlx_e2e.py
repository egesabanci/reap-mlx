"""End-to-end pruning runs on tiny random MLX-LM models.

The other suites use fake modules. These tests build the real mlx-lm
architectures (bfloat16 and 4-bit quantized) so they cover what fakes cannot:
quantized router gates, architecture-specific config keys, bfloat16 to NumPy
boundaries, mlx-lm reload, and observer replay parity against the live layers.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
import json
import random

import numpy as np
import pytest

from reap.data import load_calibration_sequences
from reap.entrypoint import main
from reap.model_adapters import config_num_experts
from reap.observer import observe_model

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None,
    reason="MLX-LM is not installed in this environment.",
)

VOCAB_WORDS = [f"w{i}" for i in range(200)]
NUM_EXPERTS = 8

FAMILY_CONFIGS = {
    "qwen3_moe": {
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "moe_intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 512,
        "norm_topk_prob": True,
    },
    "qwen2_moe": {
        "model_type": "qwen2_moe",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_experts": NUM_EXPERTS,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
    },
    "olmoe": {
        "model_type": "olmoe",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "norm_topk_prob": False,
    },
    "mixtral": {
        "model_type": "mixtral",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_key_value_heads": 2,
        "num_local_experts": NUM_EXPERTS,
    },
    "phimoe": {
        "model_type": "phimoe",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "original_max_position_embeddings": 256,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        # Widens mlx-lm 0.32's sparsemixer band so weights mix several experts.
        "router_jitter_noise": 0.3,
        "rope_scaling": {
            "short_factor": [1.0] * 8,
            "long_factor": [1.0] * 8,
            "short_mscale": 1.0,
            "long_mscale": 1.0,
            "type": "longrope",
        },
    },
    "lfm2_moe": {
        "model_type": "lfm2_moe",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "use_expert_bias": True,
        # Scales mlx-lm 0.32's sigmoid routing weights; mlx-lm 0.31 ignores it.
        "routed_scaling_factor": 1.5,
        "num_dense_layers": 1,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
    },
}
FAMILIES = sorted(FAMILY_CONFIGS)

# mlx-lm 0.32 can give every switch-layer expert its own nvfp4 global scale.
NVFP4_GLOBAL_SCALE = {
    "group_size": 16,
    "bits": 4,
    "mode": "nvfp4",
    "global_scale": True,
}
QUANTIZATIONS = pytest.mark.parametrize(
    "quantize",
    [False, True, NVFP4_GLOBAL_SCALE],
    ids=["bf16", "4bit", "nvfp4-global-scale"],
)


def _build_model(family: str, *, quantize: bool | dict):
    """Return a deterministic tiny bfloat16 model and its config.

    ``quantize`` is True for 4-bit affine, or a dict of quantize_model options.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import quantize_model

    options = {"group_size": 64, "bits": 4} if quantize is True else quantize
    supported = inspect.signature(quantize_model).parameters
    if options and not set(options) <= set(supported):
        pytest.skip(f"The installed mlx-lm cannot quantize with {options}.")

    mx.random.seed(0)
    config = copy.deepcopy(FAMILY_CONFIGS[family])
    arch = importlib.import_module(f"mlx_lm.models.{family}")
    model = arch.Model(arch.ModelArgs.from_dict(config))
    for layer in model.layers:
        for name in ("mlp", "block_sparse_moe", "feed_forward"):
            moe = getattr(layer, name, None)
            gate = getattr(moe, "gate", None)
            if isinstance(gate, nn.Linear):
                # Sharper logits make routing decisive and experts distinct.
                gate.weight = gate.weight * 8.0
            if getattr(moe, "use_expert_bias", False):
                moe.expert_bias = mx.random.uniform(-0.05, 0.05, (NUM_EXPERTS,))
    model.set_dtype(mx.bfloat16)
    if options:
        model, config = quantize_model(model, config, **options)
    mx.eval(model.parameters())
    return model, config


def _write_tokenizer(path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
    for word in VOCAB_WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="[UNK]",
    ).save_pretrained(path)


def _write_source_model(path, family: str, *, quantize: bool | dict):
    from mlx_lm.utils import save_config, save_model

    model, config = _build_model(family, quantize=quantize)
    save_model(path, model, donate_model=True)
    save_config(dict(config), path / "config.json")
    _write_tokenizer(path)
    return path


def _calibration_records(*_args, **_kwargs):
    rng = random.Random(0)
    return [
        {"text": " ".join(rng.choice(VOCAB_WORDS) for _ in range(24))}
        for _ in range(6)
    ]


@QUANTIZATIONS
@pytest.mark.parametrize("family", FAMILIES)
def test_cli_prunes_saves_and_reloads_tiny_model(tmp_path, family, quantize):
    source = _write_source_model(tmp_path / "source", family, quantize=quantize)
    output = tmp_path / "pruned"

    exit_code = main(
        [
            "--model-name",
            str(source),
            "--dataset-name",
            "tiny-synthetic",
            "--compression-ratio",
            "0.25",
            "--max-samples",
            "4",
            "--max-seq-length",
            "24",
            "--output-dir",
            str(output),
            "--smoke-max-tokens",
            "2",
            "--eval-calibration-nll",
        ],
        load_calibration_sequences_fn=functools.partial(
            load_calibration_sequences,
            load_dataset_fn=_calibration_records,
        ),
        print_fn=lambda _message: None,
    )

    assert exit_code == 0
    metrics = json.loads((output / "validation-metrics.json").read_text())
    assert metrics["status"] == "success"
    assert metrics["smoke"]["completed"] is True
    quality = metrics["quality"]
    assert quality["status"] == "ok"
    assert quality["evaluation_set"] == "held_out"
    assert quality["baseline"]["status"] == "ok"
    assert quality["delta_mean_nll"] == pytest.approx(
        quality["mean_nll"] - quality["baseline"]["mean_nll"]
    )
    # Only the observed sequences count as calibration samples.
    assert metrics["run_config"]["actual_sample_count"] == 4
    saved_config = json.loads((output / "config.json").read_text())
    assert config_num_experts(saved_config) == NUM_EXPERTS - NUM_EXPERTS // 4
    assert metrics["save_reload"]["reloaded_config_expert_count"] == 6


def test_cli_packs_calibration_into_full_length_sequences(tmp_path):
    source = _write_source_model(tmp_path / "source", "qwen3_moe", quantize=False)
    output = tmp_path / "pruned"

    exit_code = main(
        [
            "--model-name",
            str(source),
            "--dataset-name",
            "tiny-synthetic",
            "--max-samples",
            "2",
            "--max-seq-length",
            "32",
            "--pack-calibration",
            "--output-dir",
            str(output),
            "--no-smoke",
        ],
        load_calibration_sequences_fn=functools.partial(
            load_calibration_sequences,
            load_dataset_fn=_calibration_records,
        ),
        print_fn=lambda _message: None,
    )

    assert exit_code == 0
    metrics = json.loads((output / "validation-metrics.json").read_text())
    # Each 24-token sample is packed into exactly 32-token sequences.
    assert metrics["run_config"]["actual_token_counts"] == [32, 32]


@pytest.mark.parametrize("family", ["qwen2_moe", "lfm2_moe"])
def test_observer_statistics_do_not_depend_on_eval_frequency(family):
    model, config = _build_model(family, quantize=True)
    rng = np.random.default_rng(3)
    sequences = [
        {"input_ids": rng.integers(3, 200, size=length).astype(np.int32)}
        for length in (21, 40)
    ]

    every_layer, every_third = (
        observe_model(model, sequences, config, eval_frequency=frequency)
        for frequency in (1, 3)
    )

    assert every_layer.keys() == every_third.keys()
    for layer_idx, report in every_layer.items():
        for key in (
            "expert_frequency",
            "ean_sum",
            "weighted_ean_sum",
            "reap",
            "max_activations",
            "shared_expert_ean_sum",
        ):
            np.testing.assert_array_equal(report[key], every_third[layer_idx][key])


@QUANTIZATIONS
@pytest.mark.parametrize("family", FAMILIES)
def test_observer_replay_is_bit_identical_to_live_layers(family, quantize):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    model, config = _build_model(family, quantize=quantize)
    # 37 tokens * top-2 routes exercises SwitchGLU's sorted-gather path.
    token_ids = np.random.default_rng(1).integers(3, 200, size=37).astype(np.int32)
    hidden = model.model.embed_tokens(mx.array(token_ids)[None])
    attention_mask = create_attention_mask(hidden, None)
    ssm_mask = create_ssm_mask(hidden, None)
    expected = []
    for layer in model.layers:
        is_attention = getattr(layer, "is_attention_layer", True)
        hidden = layer(hidden, attention_mask if is_attention else ssm_mask, cache=None)
        mx.eval(hidden)
        expected.append(hidden)

    replayed = []

    def capture(hidden_states):
        mx.eval(hidden_states)
        replayed.append(hidden_states)

    observer_data = observe_model(
        model, [{"input_ids": token_ids}], config, eval_fn=capture
    )

    assert len(replayed) == len(expected)
    for layer_idx, (got, want) in enumerate(zip(replayed, expected)):
        assert mx.array_equal(got, want).item(), f"layer {layer_idx} diverged"
    assert observer_data
    for layer_idx, layer_data in observer_data.items():
        assert layer_data["replay_parity_error"] == 0.0, layer_idx
