"""Tests for the MLX pruning CLI entrypoint."""

from __future__ import annotations

import json
import logging
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
        "--allow-partial-calibration",
        "--output-dir",
        "out",
        *extra_args,
    ]

    with pytest.raises(SystemExit) as exc_info:
        main(args, load_model_fn=fail_load_model)

    assert exc_info.value.code == 2


def _fail_load_model(model_name):
    raise AssertionError("load_model_fn must not be called for invalid args")


def _write_fake_model_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"original-weights")


def test_output_dir_equal_to_source_model_dir_is_rejected(tmp_path):
    source = tmp_path / "local-model"
    _write_fake_model_files(source)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--model-name",
                str(source),
                "--dataset-name",
                "dataset",
                "--output-dir",
                f"{source}/",
            ],
            load_model_fn=_fail_load_model,
        )

    assert exc_info.value.code == 2
    assert (source / "model.safetensors").read_bytes() == b"original-weights"


def test_output_dir_with_foreign_model_files_is_rejected(tmp_path):
    output = tmp_path / "someone-elses-model"
    _write_fake_model_files(output)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--model-name",
                "model",
                "--dataset-name",
                "dataset",
                "--output-dir",
                str(output),
            ],
            load_model_fn=_fail_load_model,
        )

    assert exc_info.value.code == 2
    assert (output / "model.safetensors").read_bytes() == b"original-weights"


def test_output_dir_from_a_previous_reap_run_is_reused(tmp_path):
    output = tmp_path / "pruned"
    _write_fake_model_files(output)
    (output / "validation-metrics.json").write_text("{}", encoding="utf-8")

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--allow-partial-calibration",
            "--output-dir",
            str(output),
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=output
        ),
        print_fn=lambda message: None,
    )

    assert code == 0


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
            "--allow-partial-calibration",
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
        "pack": False,
        "streaming": False,
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
    # A prune checkpoint is written so a failed save can be resumed.
    assert (tmp_path / "pruned" / "reap-checkpoint.json").exists()
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
            "--allow-partial-calibration",
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
    assert observe_kwargs["verify_replay"] is True

    payload = json.loads(
        (tmp_path / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["run_config"]["eval_frequency"] == 4


def _run_with_partial_observation(tmp_path, *extra_args):
    output = []
    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--max-samples",
            "2",
            "--output-dir",
            str(tmp_path),
            "--no-smoke",
            *extra_args,
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [
            {"input_ids": [1, 2]},
            {"input_ids": [3, 4]},
        ],
        # Out of memory after the first of two sequences.
        observe_model_fn=lambda *args, **kwargs: {
            0: {"reap": [1.0, 0.5], "total_tokens": 2, "observed_sequences": 1}
        },
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=output.append,
    )
    return code, output


def test_partial_observation_must_still_meet_min_calibration_samples(tmp_path):
    with pytest.raises(RuntimeError, match="Observation stopped after 1 of 2"):
        _run_with_partial_observation(tmp_path)

    payload = json.loads(
        (tmp_path / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["observer"]["observed_sequences"] == 1


def test_partial_observation_is_allowed_with_allow_partial_calibration(tmp_path):
    code, output = _run_with_partial_observation(
        tmp_path, "--allow-partial-calibration"
    )

    assert code == 0
    assert any("only 1/2 calibration sequences" in line for line in output)


def _run_with_eval_nll(tmp_path, monkeypatch, *, available_records):
    import reap.eval_nll

    model, tokenizer, config = _moe_model_and_config()
    reloaded_model = object()
    events = []
    loader_kwargs = {}

    def fake_loader(tokenizer_arg, dataset_name, **kwargs):
        loader_kwargs.update(kwargs)
        count = min(kwargs["max_samples"], available_records)
        return [{"input_ids": [index, index + 1]} for index in range(count)]

    def fake_observe(model_arg, sequences, config_arg, **kwargs):
        events.append(("observe", [seq["input_ids"][0] for seq in sequences]))
        return {0: {"reap": [1.0, 0.5]}}

    def fake_prune(*args, **kwargs):
        events.append(("prune",))
        return {}

    def fake_nll(model_arg, sequences, **kwargs):
        which = "baseline" if model_arg is model else "pruned"
        events.append((which, [seq["input_ids"][0] for seq in sequences]))
        return {
            "status": "ok",
            "mean_nll": 2.0 if which == "baseline" else 2.5,
            "token_count": len(sequences),
            "sequence_count": len(sequences),
        }

    monkeypatch.setattr(reap.eval_nll, "calibration_mean_nll", fake_nll)
    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--max-samples",
            "3",
            "--eval-calibration-nll",
            "--eval-calibration-sequences",
            "2",
            "--output-dir",
            str(tmp_path),
            "--no-smoke",
        ],
        load_model_fn=lambda model_name: (model, tokenizer, config),
        load_calibration_sequences_fn=fake_loader,
        observe_model_fn=fake_observe,
        prune_experts_fn=fake_prune,
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path, reloaded_model=reloaded_model
        ),
        print_fn=lambda message: None,
    )
    payload = json.loads((tmp_path / "validation-metrics.json").read_text())
    return code, events, loader_kwargs, payload["quality"]


def test_eval_nll_holds_out_sequences_and_measures_a_pre_prune_baseline(
    tmp_path, monkeypatch
):
    code, events, loader_kwargs, quality = _run_with_eval_nll(
        tmp_path, monkeypatch, available_records=10
    )

    assert code == 0
    assert loader_kwargs["max_samples"] == 5
    assert events == [
        ("observe", [0, 1, 2]),
        ("baseline", [3, 4]),
        ("prune",),
        ("pruned", [3, 4]),
    ]
    assert quality["evaluation_set"] == "held_out"
    assert quality["baseline"]["mean_nll"] == 2.0
    assert quality["mean_nll"] == 2.5
    assert quality["delta_mean_nll"] == pytest.approx(0.5)


def test_eval_nll_falls_back_to_observed_sequences_when_none_are_left(
    tmp_path, monkeypatch
):
    code, events, _, quality = _run_with_eval_nll(
        tmp_path, monkeypatch, available_records=3
    )

    assert code == 0
    assert ("baseline", [0, 1]) in events
    assert quality["evaluation_set"] == "calibration"


def test_pack_calibration_flag_is_forwarded_to_the_loader(tmp_path):
    loader_kwargs = {}

    def fake_load_calibration_sequences(tokenizer_arg, dataset_name, **kwargs):
        loader_kwargs.update(kwargs)
        return [{"input_ids": [1]}]

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--allow-partial-calibration",
            "--output-dir",
            str(tmp_path),
            "--pack-calibration",
            "--dataset-streaming",
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        load_calibration_sequences_fn=fake_load_calibration_sequences,
        observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
        prune_experts_fn=lambda *args, **kwargs: {},
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert loader_kwargs["pack"] is True
    assert loader_kwargs["streaming"] is True
    payload = json.loads(
        (tmp_path / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["run_config"]["pack_calibration"] is True
    assert payload["run_config"]["dataset_streaming"] is True


def test_skip_replay_check_disables_observer_parity_verification(tmp_path):
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
            "--allow-partial-calibration",
            "--output-dir",
            str(tmp_path),
            "--skip-replay-check",
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
    assert observe_kwargs["verify_replay"] is False
    payload = json.loads(
        (tmp_path / "validation-metrics.json").read_text(encoding="utf-8")
    )
    assert payload["run_config"]["replay_check_enabled"] is False


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
            "--allow-partial-calibration",
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
            "--allow-partial-calibration",
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
            "--allow-partial-calibration",
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
                "--allow-partial-calibration",
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


def _multi_moe_model_and_config(num_layers: int = 2, num_experts: int = 4):
    """Return a multi-layer MoE model for selective-prune CLI tests."""
    layers = []
    for _ in range(num_layers):
        layers.append(
            SimpleNamespace(
                mlp=SimpleNamespace(
                    switch_mlp=object(),
                    num_experts=num_experts,
                    top_k=2,
                    num_experts_per_tok=2,
                )
            )
        )
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    config = {
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "hidden_size": 64,
        "num_hidden_layers": num_layers,
    }
    return model, object(), config


def test_main_passes_layer_selection_flags_to_prune(tmp_path):
    prune_kwargs = {}

    def fake_prune_experts(model_arg, config_arg, observer_data, method, ratio, **kwargs):
        del model_arg, config_arg, observer_data, method, ratio
        prune_kwargs.update(kwargs)
        return {0: [0, 1, 2, 3]}

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--allow-partial-calibration",
            "--output-dir",
            str(tmp_path),
            # ratio 0 keeps uniform expert counts when skipping a layer
            "--compression-ratio",
            "0",
            "--prune-layer-indices",
            "0",
            "--skip-layer-indices",
            "1",
        ],
        load_model_fn=lambda model_name: (*_multi_moe_model_and_config(2),),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {
            0: {"reap": [1.0, 0.5, 0.25, 0.1], "total_tokens": 1},
            1: {"reap": [1.0, 0.5, 0.25, 0.1], "total_tokens": 1},
        },
        prune_experts_fn=fake_prune_experts,
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert prune_kwargs["prune_layer_indices"] == [0]
    assert prune_kwargs["skip_layer_indices"] == [1]


def test_main_passes_per_layer_ratios_to_prune(tmp_path):
    prune_kwargs = {}

    def fake_prune_experts(model_arg, config_arg, observer_data, method, ratio, **kwargs):
        del model_arg, config_arg, observer_data, method, ratio
        prune_kwargs.update(kwargs)
        return {0: [0, 1], 1: [0, 1]}

    # Heterogeneous experts: 4 * 0.5 -> 2 retained, 8 * 0.75 -> 2 retained.
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        switch_mlp=object(),
                        num_experts=4,
                        top_k=2,
                        num_experts_per_tok=2,
                    )
                ),
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        switch_mlp=object(),
                        num_experts=8,
                        top_k=2,
                        num_experts_per_tok=2,
                    )
                ),
            ]
        )
    )
    config = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_size": 64,
        "num_hidden_layers": 2,
    }

    code = main(
        [
            "--model-name",
            "model",
            "--dataset-name",
            "dataset",
            "--allow-partial-calibration",
            "--output-dir",
            str(tmp_path),
            "--per-layer-ratios",
            "0:0.5",
            "1:0.75",
        ],
        load_model_fn=lambda model_name: (model, object(), config),
        load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
        observe_model_fn=lambda *args, **kwargs: {
            0: {"reap": [1.0, 0.5, 0.25, 0.1], "total_tokens": 1},
            1: {
                "reap": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                "total_tokens": 1,
            },
        },
        prune_experts_fn=fake_prune_experts,
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path
        ),
        print_fn=lambda message: None,
    )

    assert code == 0
    assert prune_kwargs["per_layer_ratios"] == {0: 0.5, 1: 0.75}


def test_main_rejects_malformed_per_layer_ratios(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--model-name",
                "model",
                "--dataset-name",
                "dataset",
                "--allow-partial-calibration",
                "--output-dir",
                str(tmp_path),
                "--per-layer-ratios",
                "0-0.5",
            ],
            load_model_fn=lambda model_name: (*_moe_model_and_config(),),
            load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
            observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
            prune_experts_fn=lambda *args, **kwargs: {},
            save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
                output_dir=tmp_path
            ),
            print_fn=lambda message: None,
        )
    # argparse parser.error exits with code 2 on bad --per-layer-ratios format.
    assert exc_info.value.code == 2


def test_main_resume_skips_observe_and_prune_and_applies_checkpoint(tmp_path):
    import numpy as np
    from reap.checkpoint import write_checkpoint

    ckpt_path = tmp_path / "reap-checkpoint.json"
    write_checkpoint(
        ckpt_path,
        keep_by_layer={0: np.array([1, 3])},
        config_before_prune={"num_experts": 4, "top_k": 3},
        model_name="model",
        prune_method="reap",
        compression_ratio=0.5,
        adapter_name="qwen3_moe",
    )

    events: list[str] = []
    apply_calls: list[dict[int, list[int]]] = []

    def fake_observe(*args, **kwargs):
        events.append("observe")
        return {0: {"reap": [1.0]}}

    def fake_prune(*args, **kwargs):
        events.append("prune")
        return {0: [0, 1]}

    def fake_apply(model_arg, config_arg, keep_by_layer, **kwargs):
        del model_arg, config_arg, kwargs
        apply_calls.append({int(k): list(v) for k, v in keep_by_layer.items()})
        return {int(k): np.asarray(v) for k, v in keep_by_layer.items()}

    def fake_save(*args, **kwargs):
        events.append("save")
        return SimpleNamespace(output_dir=tmp_path)

    code = main(
        [
            "--model-name",
            "model",
            "--output-dir",
            str(tmp_path),
            "--resume-from-checkpoint",
            str(ckpt_path),
        ],
        load_model_fn=lambda model_name: (*_moe_model_and_config(),),
        observe_model_fn=fake_observe,
        prune_experts_fn=fake_prune,
        apply_keep_indices_fn=fake_apply,
        save_pruned_model_fn=fake_save,
        print_fn=lambda message: None,
    )

    assert code == 0
    # Resume must skip observation and pruning, and only run save.
    assert "observe" not in events
    assert "prune" not in events
    assert "save" in events
    # The checkpoint keep indices were forwarded to apply_keep_indices.
    assert apply_calls == [{0: [1, 3]}]


def test_checkpoint_is_written_before_pruning_mutates_the_model(tmp_path):
    import numpy as np

    def prune_then_fail(*args, on_plan=None, **kwargs):
        on_plan({0: np.array([0, 2])})
        raise RuntimeError("slicing failed")

    with pytest.raises(RuntimeError, match="slicing failed"):
        main(
            [
                "--model-name",
                "model",
                "--dataset-name",
                "dataset",
                "--allow-partial-calibration",
                "--output-dir",
                str(tmp_path),
            ],
            load_model_fn=lambda model_name: (*_moe_model_and_config(),),
            load_calibration_sequences_fn=lambda *args, **kwargs: [{"input_ids": [1]}],
            observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0, 0.5, 0.2]}},
            prune_experts_fn=prune_then_fail,
            print_fn=lambda message: None,
        )

    checkpoint = json.loads((tmp_path / "reap-checkpoint.json").read_text())
    assert checkpoint["keep_by_layer"] == {"0": [0, 2]}
    payload = json.loads((tmp_path / "validation-metrics.json").read_text())
    assert payload["failure"]["phase"] == "prune"


def _write_resume_checkpoint(tmp_path, *, model_name="model", config_before_prune=None):
    import numpy as np
    from reap.checkpoint import write_checkpoint

    return write_checkpoint(
        tmp_path / "checkpoint" / "reap-checkpoint.json",
        keep_by_layer={0: np.array([1, 3])},
        config_before_prune=config_before_prune or {"num_experts": 4},
        model_name=model_name,
        prune_method="reap",
        compression_ratio=0.5,
        adapter_name="qwen3_moe",
    )


def _resume(tmp_path, model_name, checkpoint_path, *extra_args):
    applied = []

    def fake_apply(model_arg, config_arg, keep_by_layer, **kwargs):
        applied.append(keep_by_layer)
        return dict(keep_by_layer)

    code = main(
        [
            "--model-name",
            model_name,
            "--output-dir",
            str(tmp_path / "out"),
            "--resume-from-checkpoint",
            str(checkpoint_path),
            "--no-smoke",
            *extra_args,
        ],
        load_model_fn=lambda name: (*_moe_model_and_config(),),
        apply_keep_indices_fn=fake_apply,
        save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
            output_dir=tmp_path / "out"
        ),
        print_fn=lambda message: None,
    )
    return code, applied


def test_resume_records_the_checkpoint_plan_instead_of_cli_defaults(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path)

    code, _ = _resume(tmp_path, "model", checkpoint)

    assert code == 0
    payload = json.loads((tmp_path / "out" / "validation-metrics.json").read_text())
    assert payload["run_config"]["compression_ratio"] == 0.5
    assert payload["run_config"]["checkpoint"]["path"] == str(checkpoint)
    assert payload["run_config"]["resume_from_checkpoint"] == str(checkpoint)


def test_default_loader_passes_revision_to_mlx_lm(monkeypatch):
    import mlx_lm

    from reap.entrypoint import _default_load_model

    calls = []

    def fake_load(model_name, return_config=False, revision=None):
        calls.append((model_name, return_config, revision))
        return object(), object(), {"model_type": "qwen3_moe"}

    monkeypatch.setattr(mlx_lm, "load", fake_load)

    _default_load_model("org/model", revision="abc123")

    assert calls == [("org/model", True, "abc123")]


def test_resolve_hub_revision_reads_the_cached_snapshot_commit(tmp_path, monkeypatch):
    import huggingface_hub

    from reap.entrypoint import _resolve_hub_revision

    snapshot = tmp_path / "snapshots" / "0123abcd"
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, revision=None, local_files_only=False: str(snapshot),
    )

    assert _resolve_hub_revision("org/model", "main") == "0123abcd"
    # Local paths have no Hub revision.
    assert _resolve_hub_revision(str(tmp_path), None) is None


def test_resolve_hub_revision_falls_back_to_the_requested_revision(monkeypatch):
    import huggingface_hub

    from reap.entrypoint import _resolve_hub_revision

    def not_cached(repo_id, revision=None, local_files_only=False):
        raise FileNotFoundError(repo_id)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)

    assert _resolve_hub_revision("org/model", "v2") == "v2"
    assert _resolve_hub_revision("org/model", None) is None


def test_resume_accepts_another_spelling_of_the_same_local_model_path(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    checkpoint = _write_resume_checkpoint(tmp_path, model_name=str(model_dir))

    code, applied = _resume(tmp_path, f"{model_dir}/", checkpoint)

    assert code == 0
    assert applied


def test_resume_rejects_a_checkpoint_made_for_a_different_layout(tmp_path):
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config_before_prune={"num_experts": 8, "num_hidden_layers": 1},
    )

    with pytest.raises(ValueError, match="made for 8 experts per layer"):
        _resume(tmp_path, "model", checkpoint)


def test_no_strict_resume_downgrades_layout_mismatch_to_a_warning(tmp_path, caplog):
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config_before_prune={"num_experts": 8},
    )

    with caplog.at_level(logging.WARNING, logger="reap.entrypoint"):
        code, applied = _resume(tmp_path, "model", checkpoint, "--no-strict-resume")

    assert code == 0
    assert applied
    assert any("made for 8 experts" in record.message for record in caplog.records)


def test_strict_resume_flag_is_accepted_explicitly(tmp_path):
    checkpoint = _write_resume_checkpoint(tmp_path)

    code, applied = _resume(tmp_path, "model", checkpoint, "--strict-resume")

    assert code == 0
    assert applied


def test_main_resume_requires_existing_checkpoint_file(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--model-name",
                "model",
                "--output-dir",
                str(tmp_path),
                "--resume-from-checkpoint",
                str(tmp_path / "does-not-exist.json"),
            ],
            load_model_fn=lambda model_name: (*_moe_model_and_config(),),
            observe_model_fn=lambda *args, **kwargs: {0: {"reap": [1.0]}},
            prune_experts_fn=lambda *args, **kwargs: {},
            save_pruned_model_fn=lambda *args, **kwargs: SimpleNamespace(
                output_dir=tmp_path
            ),
            print_fn=lambda message: None,
        )
    assert exc_info.value.code == 2
