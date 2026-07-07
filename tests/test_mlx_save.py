"""Tests for MLX save/reload validation helpers."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reap.save import (
    SaveReloadResult,
    generation_smoke,
    save_pruned_model,
)


MLX_LM_AVAILABLE = importlib.util.find_spec("mlx_lm") is not None


def test_save_module_import_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during MLX save import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.save import SaveReloadResult, save_pruned_model

        assert SaveReloadResult is not None
        assert save_pruned_model is not None

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX save import: "
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


class ConfigTrapModel:
    """Mock model with minimal MoE structure for adapter inference."""
    @property
    def config(self):
        raise AssertionError("save helper must not read model.config")

# Give it MoE-like structure so infer_model_adapter recognizes it
ConfigTrapModel.model = SimpleNamespace(
    layers=[SimpleNamespace(mlp=SimpleNamespace(switch_mlp=object()))],
)
class TinyLinear:
    def __init__(self, num_experts: int):
        self.weight = np.zeros((num_experts, 2), dtype=np.float32)


class TinySwitchMlp:
    def __init__(self, num_experts: int):
        self.gate_proj = TinyLinear(num_experts)
        self.up_proj = TinyLinear(num_experts)
        self.down_proj = TinyLinear(num_experts)


class TinyGate:
    def __init__(self, num_experts: int):
        self.weight = np.zeros((num_experts, 2), dtype=np.float32)


class TinyMoe:
    def __init__(self, num_experts: int, *, weight_experts: int | None = None):
        self.num_experts = num_experts
        self.top_k = min(2, num_experts)
        weight_experts = num_experts if weight_experts is None else weight_experts
        self.switch_mlp = TinySwitchMlp(weight_experts)
        self.gate = TinyGate(weight_experts)


class TinyLfmMoe(TinyMoe):
    def __init__(
        self,
        num_experts: int,
        *,
        weight_experts: int | None = None,
        expert_bias_experts: int | None = None,
    ):
        super().__init__(num_experts, weight_experts=weight_experts)
        self.use_expert_bias = True
        expert_bias_experts = (
            num_experts if expert_bias_experts is None else expert_bias_experts
        )
        self.expert_bias = np.zeros((expert_bias_experts,), dtype=np.float32)


def make_reloaded_model(num_experts: int, *, weight_experts: int | None = None):
    moe = TinyMoe(num_experts, weight_experts=weight_experts)
    return SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]),
    )


def make_lfm2_reloaded_model(
    num_experts: int,
    *,
    weight_experts: int | None = None,
    expert_bias_experts: int | None = None,
):
    moe = TinyLfmMoe(
        num_experts,
        weight_experts=weight_experts,
        expert_bias_experts=expert_bias_experts,
    )
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(feed_forward=object()),
                SimpleNamespace(feed_forward=moe),
            ],
        ),
    )


def write_required_artifacts(
    output_dir: str | Path,
    *,
    config: dict | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "config.json").write_text(
        json.dumps({} if config is None else config),
        encoding="utf-8",
    )
    (output_path / "model.safetensors").write_bytes(b"weights")
    (output_path / "tokenizer.json").write_bytes(b"{}")


def fake_save_factory(calls, *, write_artifacts=True, save_passed_config=False):
    def fake_save(**kwargs):
        calls.append(kwargs)
        if write_artifacts:
            write_required_artifacts(
                kwargs["dst_path"],
                config=kwargs["config"] if save_passed_config else None,
            )

    return fake_save


def fake_load_factory(model, tokenizer, config, calls=None):
    def fake_load(path, *, return_config=False):
        if calls is not None:
            calls.append({"path": path, "return_config": return_config})
        assert return_config is True
        return model, tokenizer, config

    return fake_load


def fake_load_without_return_config_factory(model, tokenizer, calls=None):
    def fake_load(path):
        if calls is not None:
            calls.append({"path": path})
        return model, tokenizer

    return fake_load


def test_save_pruned_model_uses_passed_config_and_validates_reload(tmp_path):
    save_calls = []
    load_calls = []
    original_model = ConfigTrapModel()
    original_tokenizer = object()
    reloaded_model = make_reloaded_model(2)
    reloaded_tokenizer = object()
    config = {"num_experts": 2, "num_experts_per_tok": 2}

    result = save_pruned_model(
        original_model,
        original_tokenizer,
        config,
        tmp_path / "saved",
        "source-model",
        save_fn=fake_save_factory(save_calls),
        load_fn=fake_load_factory(
            reloaded_model,
            reloaded_tokenizer,
            {"num_experts": 2, "num_experts_per_tok": 2},
            calls=load_calls,
        ),
    )

    assert isinstance(result, SaveReloadResult)
    assert result.output_dir == tmp_path / "saved"
    assert result.reloaded_model is reloaded_model
    assert result.reloaded_tokenizer is reloaded_tokenizer
    assert result.reloaded_config == {"num_experts": 2, "num_experts_per_tok": 2}
    assert result.expected_expert_count == 2
    assert result.smoke_result is None

    assert len(save_calls) == 1
    assert save_calls[0]["dst_path"] == str(tmp_path / "saved")
    assert save_calls[0]["src_path_or_repo"] == "source-model"
    assert save_calls[0]["model"] is original_model
    assert save_calls[0]["tokenizer"] is original_tokenizer
    assert save_calls[0]["config"] is config
    assert load_calls == [
        {"path": str(tmp_path / "saved"), "return_config": True},
    ]


def test_save_pruned_model_supports_load_without_return_config(tmp_path):
    load_calls = []
    reloaded_model = make_reloaded_model(2)
    reloaded_tokenizer = object()
    config = {"num_experts": 2, "num_experts_per_tok": 2}

    result = save_pruned_model(
        object(),
        object(),
        config,
        tmp_path / "saved",
        "source-model",
        save_fn=fake_save_factory([], save_passed_config=True),
        load_fn=fake_load_without_return_config_factory(
            reloaded_model,
            reloaded_tokenizer,
            calls=load_calls,
        ),
    )

    assert result.reloaded_model is reloaded_model
    assert result.reloaded_tokenizer is reloaded_tokenizer
    assert result.reloaded_config == config
    assert load_calls == [{"path": str(tmp_path / "saved")}]


def test_smoke_function_runs_on_reloaded_model_not_original(tmp_path):
    smoke_calls = []
    original_model = object()
    reloaded_model = make_reloaded_model(1)
    reloaded_tokenizer = object()

    def smoke_fn(model, tokenizer, config):
        smoke_calls.append((model, tokenizer, config))
        return "smoke-ok"

    result = save_pruned_model(
        original_model,
        object(),
        {"num_experts": 1},
        tmp_path / "saved",
        "source-model",
        smoke_fn=smoke_fn,
        save_fn=fake_save_factory([]),
        load_fn=fake_load_factory(
            reloaded_model,
            reloaded_tokenizer,
            {"num_experts": 1},
        ),
    )

    assert result.smoke_result == "smoke-ok"
    assert smoke_calls == [(reloaded_model, reloaded_tokenizer, {"num_experts": 1})]


def test_save_pruned_model_records_configured_smoke_prompt_and_max_tokens(tmp_path):
    result = save_pruned_model(
        object(),
        object(),
        {"num_experts": 1},
        tmp_path / "saved",
        "source-model",
        smoke_fn=lambda model, tokenizer, config: "smoke-ok",
        smoke_prompt="Summarize this domain.",
        smoke_max_tokens=7,
        save_fn=fake_save_factory([]),
        load_fn=fake_load_factory(
            make_reloaded_model(1),
            object(),
            {"num_experts": 1},
        ),
    )

    assert result.metrics["smoke"]["prompt"] == "Summarize this domain."
    assert result.metrics["smoke"]["max_tokens"] == 7
    assert result.metrics["smoke"]["completed"] is True


def test_save_pruned_model_uses_explicit_expected_count_when_config_missing(tmp_path):
    result = save_pruned_model(
        object(),
        object(),
        {},
        tmp_path / "saved",
        "source-model",
        expected_expert_count=3,
        save_fn=fake_save_factory([]),
        load_fn=fake_load_factory(
            make_reloaded_model(3),
            object(),
            {"num_experts": 3},
        ),
    )

    assert result.expected_expert_count == 3


@pytest.mark.skipif(
    MLX_LM_AVAILABLE,
    reason="mlx_lm is installed, so the lazy missing-dependency path is unavailable.",
)
def test_save_pruned_model_requires_lazy_mlx_lm_without_injected_functions(tmp_path):
    with pytest.raises(ModuleNotFoundError, match="mlx_lm"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 1},
            tmp_path / "saved",
            "source-model",
        )


def test_save_pruned_model_rejects_missing_expected_expert_count(tmp_path):
    with pytest.raises(ValueError, match="expected_expert_count"):
        save_pruned_model(
            object(),
            object(),
            {},
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=fake_load_factory(make_reloaded_model(1), object(), {}),
        )


def test_save_pruned_model_rejects_missing_saved_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="missing config.json"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 1},
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([], write_artifacts=False),
            load_fn=fake_load_factory(make_reloaded_model(1), object(), {}),
        )


def test_save_pruned_model_rejects_invalid_reload_return(tmp_path):
    def bad_load(path, *, return_config=False):
        assert return_config is True
        return make_reloaded_model(1), object()

    with pytest.raises(ValueError, match="return_config=True"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 1},
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=bad_load,
        )


def test_save_pruned_model_rejects_reload_config_mismatch(tmp_path):
    with pytest.raises(ValueError, match="Reloaded config expert count mismatch"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 2},
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=fake_load_factory(
                make_reloaded_model(2),
                object(),
                {"num_experts": 3},
            ),
        )


def test_save_pruned_model_rejects_reloaded_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="first dimension mismatch"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 2},
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=fake_load_factory(
                make_reloaded_model(2, weight_experts=3),
                object(),
                {"num_experts": 2},
            ),
        )


def test_save_pruned_model_validates_lfm2_expert_bias_shape(tmp_path):
    result = save_pruned_model(
        object(),
        object(),
        {
            "model_type": "lfm2_moe",
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "use_expert_bias": True,
        },
        tmp_path / "saved",
        "source-model",
        save_fn=fake_save_factory([]),
        load_fn=fake_load_factory(
            make_lfm2_reloaded_model(16),
            object(),
            {
                "model_type": "lfm2_moe",
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "use_expert_bias": True,
            },
        ),
    )

    assert result.expected_expert_count == 16


def test_save_pruned_model_rejects_lfm2_expert_bias_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="expert_bias first dimension mismatch"):
        save_pruned_model(
            object(),
            object(),
            {
                "model_type": "lfm2_moe",
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "use_expert_bias": True,
            },
            tmp_path / "saved",
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=fake_load_factory(
                make_lfm2_reloaded_model(16, expert_bias_experts=32),
                object(),
                {
                    "model_type": "lfm2_moe",
                    "num_experts": 16,
                    "num_experts_per_tok": 4,
                    "use_expert_bias": True,
                },
            ),
        )


def test_save_pruned_model_rejects_output_path_that_is_file(tmp_path):
    output_file = tmp_path / "saved"
    output_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError, match="not a directory"):
        save_pruned_model(
            object(),
            object(),
            {"num_experts": 1},
            output_file,
            "source-model",
            save_fn=fake_save_factory([]),
            load_fn=fake_load_factory(make_reloaded_model(1), object(), {}),
        )


def test_generation_smoke_formats_chat_prompt_and_calls_generate():
    calls = []

    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, *, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Who are you?"}]
            assert add_generation_prompt is True
            return "<chat>Who are you?</chat>"

    def generate_fn(model, tokenizer, *, prompt, max_tokens):
        calls.append((model, tokenizer, prompt, max_tokens))
        return "hello"

    model = object()
    tokenizer = Tokenizer()

    result = generation_smoke(
        model,
        tokenizer,
        prompt="Who are you?",
        max_tokens=4,
        generate_fn=generate_fn,
    )

    assert result == "hello"
    assert calls == [(model, tokenizer, "<chat>Who are you?</chat>", 4)]


def test_generation_smoke_falls_back_to_raw_prompt_when_chat_template_fails(caplog):
    calls = []

    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, *, add_generation_prompt):
            del messages, add_generation_prompt
            raise RuntimeError("broken template")

    def generate_fn(model, tokenizer, *, prompt, max_tokens):
        calls.append((model, tokenizer, prompt, max_tokens))
        return "hello"

    model = object()
    tokenizer = Tokenizer()
    caplog.set_level(logging.WARNING, logger="reap.save")

    result = generation_smoke(
        model,
        tokenizer,
        prompt="Who are you?",
        max_tokens=4,
        generate_fn=generate_fn,
    )

    assert result == "hello"
    assert calls == [(model, tokenizer, "Who are you?", 4)]
    assert "Chat template application failed" in caplog.text
