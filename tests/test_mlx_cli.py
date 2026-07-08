"""Tests for the MLX pruning CLI entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from reap.entrypoint import main


def _moe_model_and_config():
    """Return (model, config) tuple with minimal MoE structure for pipeline tests."""
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[]),
    )
    mock_mlp = SimpleNamespace(switch_mlp=object())
    mock_layer = SimpleNamespace(mlp=mock_mlp)
    model.model.layers = [mock_layer]
    config = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_size": 64,
        "num_hidden_layers": 1,
    }
    return model, object(), config


def _subprocess_with_import_blocker(body: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    code = textwrap.dedent(
        f"""
        import sys

        BLOCKED_ROOTS = ("torch", "vllm", "mlx", "mlx_lm", "datasets")

        def is_blocked(fullname):
            return any(
                fullname == root or fullname.startswith(root + ".")
                for root in BLOCKED_ROOTS
            )

        class ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if is_blocked(fullname):
                    raise AssertionError(
                        "forbidden import during MLX CLI import/run: "
                        f"{{fullname}}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        {body}
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_entrypoint_import_does_not_import_heavy_runtime_packages():
    result = _subprocess_with_import_blocker(
        """
        from reap.entrypoint import build_parser, main

        assert build_parser is not None
        assert main is not None

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX CLI import: "
                + ", ".join(forbidden_loaded)
            )
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_help_works_without_heavy_runtime_imports():
    result = _subprocess_with_import_blocker(
        """
        from reap.entrypoint import main

        raise SystemExit(main(["--help"]))
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "--model-name" in result.stdout
    assert "--dataset-name" in result.stdout
    assert "--eval-frequency" in result.stdout
    assert "--smoke-prompt" in result.stdout
    assert "--smoke-max-tokens" in result.stdout


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--compression-ratio", "1.0"], "compression_ratio"),
        (["--compression-ratio", "-0.1"], "compression_ratio"),
        (["--compression-ratio", "nan"], "compression_ratio"),
        (["--prune-method", "ean_ca"], "Unsupported prune method"),
        (["--max-samples", "0"], "max_samples"),
        (["--max-seq-length", "0"], "max_seq_length"),
        (["--eval-frequency", "0"], "eval_frequency"),
        (["--smoke-max-tokens", "0"], "smoke_max_tokens"),
    ],
)
def test_invalid_arguments_fail_before_pipeline_functions(extra_args, message):
    def fail_load_model(model_name):
        raise AssertionError("load_model_fn must not be called for invalid args")

    args = [
        "--model-name",
        "model",
        "--dataset-name",
        "dataset",
        "--output-dir",
        "out",
        *extra_args,
    ]

    with pytest.raises(SystemExit) as exc_info:
        main(args, load_model_fn=fail_load_model)

    assert exc_info.value.code == 2


def test_main_runs_pipeline_with_injected_functions_and_progress_messages(tmp_path):
    from types import SimpleNamespace
    events = []
    output = []
    mock_switch_mlp = object()
    mock_mlp = SimpleNamespace(switch_mlp=mock_switch_mlp)
    mock_layer = SimpleNamespace(mlp=mock_mlp)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[mock_layer]),
    )
    tokenizer = object()
    config = {
        "model_type": "qwen2_moe",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_size": 64,
        "num_hidden_layers": 1,
    }
    smoke = object()

    def fake_load_model(model_name):
        events.append(("load", model_name))
        return model, tokenizer, config

    def fake_load_calibration_sequences(tokenizer_arg, dataset_name, **kwargs):
        events.append(("calibrate", tokenizer_arg, dataset_name, kwargs))
        return [{"input_ids": [1, 2, 3]}]

    def fake_observe_model(model_arg, sequences, config_arg, **kwargs):
        events.append(("observe", model_arg, sequences, config_arg, kwargs))
        return {0: {"reap": [1.0, 0.5, 0.25, 0.0]}}

    def fake_prune_experts(model_arg, config_arg, observer_data, method, ratio, **kwargs):
        events.append(("prune", model_arg, config_arg, observer_data, method, ratio))
        config_arg["num_experts"] = 3
        return {0: [0, 1, 2]}

    def fake_save_pruned_model(
        model_arg,
        tokenizer_arg,
        config_arg,
        output_dir,
        original_model_name,
        **kwargs,
    ):
        events.append(
            (
                "save",
                model_arg,
                tokenizer_arg,
                config_arg,
                output_dir,
                original_model_name,
                kwargs,
            )
        )
        return SimpleNamespace(output_dir=Path(output_dir))

    code = main(
        [
            "--model-name",
            "mlx-model",
            "--dataset-name",
            "calibration-data",
            "--split",
            "validation",
            "--dataset-config-name",
            "code",
            "--prune-method",
            "frequency",
            "--compression-ratio",
            "0.25",
            "--num-calibration-sequences",
            "3",
            "--max-seq-length",
            "16",
            "--seed",
            "9",
            "--output-dir",
            str(tmp_path / "pruned"),
        ],
        load_model_fn=fake_load_model,
        load_calibration_sequences_fn=fake_load_calibration_sequences,
        observe_model_fn=fake_observe_model,
        prune_experts_fn=fake_prune_experts,
        save_pruned_model_fn=fake_save_pruned_model,
        smoke_fn=smoke,
        print_fn=output.append,
    )

    assert code == 0
    assert [event[0] for event in events] == [
        "load",
        "calibrate",
        "observe",
        "prune",
        "save",
    ]
    assert events[0] == ("load", "mlx-model")
    assert events[1][1] is tokenizer
    assert events[1][2] == "calibration-data"
    assert events[1][3] == {
        "split": "validation",
        "dataset_config_name": "code",
        "max_samples": 3,
        "max_seq_length": 16,
        "seed": 9,
    }
    assert events[2][4]["eval_frequency"] == 1
    assert "print_fn" in events[2][4]
    assert events[3][4:] == ("frequency", 0.25)
    assert events[4][6]["smoke_fn"] is smoke
    assert events[4][6]["smoke_prompt"] == "What is your name?"
    assert events[4][6]["smoke_max_tokens"] == 16
    assert any("load:" in line for line in output)
    assert any("calibrate:" in line for line in output)
    assert any("observe:" in line for line in output)
    assert any("prune:" in line for line in output)
    assert any("save:" in line for line in output)
    assert any("reload/smoke:" in line for line in output)
    assert any("done:" in line for line in output)

    metrics_path = tmp_path / "pruned" / "validation-metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["run_config"]["model_name"] == "mlx-model"
    assert payload["run_config"]["eval_frequency"] == 1
    assert payload["run_config"]["actual_sample_count"] == 1
    assert payload["pruning"]["expert_count_before"] == 4
    assert payload["pruning"]["expert_count_after"] == 3


def test_main_passes_configured_eval_frequency_to_observer(tmp_path):
    observe_kwargs = {}

    def fake_observe_model(model_arg, sequences, config_arg, **kwargs):
        del model_arg, sequences, config_arg
        observe_kwargs.update(kwargs)
        return {0: {"reap": [1.0]}}

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--eval-frequency",
            "4",
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=fake_observe_model,
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert observe_kwargs["eval_frequency"] == 4
    assert "print_fn" in observe_kwargs

    payload = json.loads(
        (tmp_path / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["run_config"]["eval_frequency"] == 4


def test_main_configures_default_generation_smoke_from_cli(tmp_path, monkeypatch):
    save_kwargs = {}
    generation_calls = []

    def fake_generation_smoke(model, tokenizer, config, *, prompt, max_tokens):
        generation_calls.append((model, tokenizer, config, prompt, max_tokens))
        return "smoke-ok"

    def fake_save_pruned_model(*args, **kwargs):
        del args
        save_kwargs.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path)

    monkeypatch.setattr("reap.entrypoint.generation_smoke", fake_generation_smoke)

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--smoke-prompt",
            "Summarize this domain.",
            "--smoke-max-tokens",
            "7",
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=fake_save_pruned_model,
        print_fn=lambda message: None,
    )

    assert code == 0
    assert save_kwargs["smoke_prompt"] == "Summarize this domain."
    assert save_kwargs["smoke_max_tokens"] == 7

    result = save_kwargs["smoke_fn"](
        "reloaded-model",
        "reloaded-tokenizer",
        {"num_experts": 1},
    )

    assert result == "smoke-ok"
    assert generation_calls == [
        (
            "reloaded-model",
            "reloaded-tokenizer",
            {"num_experts": 1},
            "Summarize this domain.",
            7,
        )
    ]


def test_no_smoke_passes_no_smoke_function_to_save(tmp_path):
    save_kwargs = {}

    def fake_save_pruned_model(*args, **kwargs):
        del args
        save_kwargs.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path)

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--no-smoke",
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=fake_save_pruned_model,
        smoke_fn=object(),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert save_kwargs["smoke_fn"] is None
    assert save_kwargs["smoke_prompt"] == "What is your name?"
    assert save_kwargs["smoke_max_tokens"] == 16


def test_keyboard_interrupt_returns_130_without_success_message():
    output = []

    def interrupt(model_name):
        del model_name
        raise KeyboardInterrupt

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--output-dir",
            "out",
        ],
        load_model_fn=interrupt,
        print_fn=output.append,
    )

    assert code == 130
    assert any("interrupted:" in line for line in output)
    assert not any("done:" in line for line in output)


def test_main_writes_failed_metrics_when_pipeline_phase_raises(tmp_path):
    output_dir = tmp_path / "failed-run"

    def fail_observe(model, sequences, config, **kwargs):
        del model, sequences, config, kwargs
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        main(
            [
                "--model-name",
                "model",
                "--dataset-name",
                "dataset",
                "--output-dir",
                str(output_dir),
            ],
            load_model_fn=lambda model_name: (*_moe_model_and_config(),),
            load_calibration_sequences_fn=lambda *args, **kwargs: [
                {"input_ids": [1, 2]},
            ],
            observe_model_fn=fail_observe,
            print_fn=lambda message: None,
        )

    payload = json.loads(
        (output_dir / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["phase"] == "observe"
    assert payload["failure"]["type"] == "RuntimeError"
    assert payload["failure"]["message"] == "observer failed"


def test_main_passes_layer_selection_flags_to_prune(tmp_path):
    prune_kwargs = {}

    def fake_prune_experts(model_arg, config_arg, observer_data, method, ratio, **kwargs):
        del model_arg, config_arg, observer_data, method, ratio
        prune_kwargs.update(kwargs)
        return {0: [0, 1, 2]}

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--prune-layer-indices",
            "0",
            "--skip-layer-indices",
            "3",
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
        prune_experts_fn=fake_prune_experts,
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert prune_kwargs["prune_layer_indices"] == [0]
    assert prune_kwargs["skip_layer_indices"] == [3]
