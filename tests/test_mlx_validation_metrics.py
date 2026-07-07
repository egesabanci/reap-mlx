"""Tests for MLX validation metrics telemetry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reap.validation_metrics import RunMetrics


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
                        "forbidden import during MLX metrics import: "
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


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds: float):
        self.value += seconds


class TinyMoe:
    switch_mlp = object()
    num_experts = 4
    top_k = 2
    norm_topk_prob = True


def test_validation_metrics_import_does_not_import_heavy_runtime_packages():
    result = _subprocess_with_import_blocker(
        """
        from reap.validation_metrics import RunMetrics

        assert RunMetrics is not None

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

    assert result.returncode == 0, result.stderr + result.stdout


def test_run_metrics_writes_success_json_with_timings_and_throughput(tmp_path):
    clock = FakeClock()
    metrics = RunMetrics(tmp_path, clock=clock)

    args = SimpleNamespace(
        model_name="model",
        dataset_name="dataset",
        dataset_config_name=None,
        split="train",
        seed=42,
        max_samples=2,
        max_seq_length=8,
        eval_frequency=1,
        prune_method="reap",
        compression_ratio=0.25,
        output_dir=str(tmp_path),
        metrics_file="validation-metrics.json",
        no_smoke=False,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=TinyMoe()),
                SimpleNamespace(mlp=object()),
            ],
        ),
    )
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
    }

    metrics.record_run_config(args)
    metrics.record_model_metadata(model, config)

    with metrics.phase("calibration"):
        clock.advance(2.0)
    calibration_sequences = [{"input_ids": np.array([1, 2, 3])}, {"input_ids": [4]}]
    metrics.record_calibration(calibration_sequences)

    observer_data = {
        0: {
            "total_tokens": 4,
            "expert_frequency": np.array([2, 1, 1, 0]),
            "reap": np.array([1.0, np.nan, np.inf, 0.5]),
        }
    }
    with metrics.phase("observe"):
        clock.advance(4.0)
    metrics.record_observer(observer_data, "reap")

    with metrics.phase("prune"):
        clock.advance(1.0)
    metrics.record_pruning(
        {0: [0, 1, 3]},
        config_before=config,
        config_after={"num_experts": 3, "num_experts_per_tok": 2},
        observer_data=observer_data,
    )

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    metrics.record_save_reload(
        SimpleNamespace(
            output_dir=tmp_path,
            reloaded_config={"num_experts": 3},
            expected_expert_count=3,
            smoke_result="hello",
            metrics={
                "timings": {
                    "save_seconds": 0.5,
                    "reload_seconds": 0.25,
                    "smoke_seconds": 0.1,
                },
                "artifacts": {
                    "file_count": 2,
                    "total_bytes": 9,
                    "files": [],
                },
                "smoke": {
                    "enabled": True,
                    "completed": True,
                    "elapsed_seconds": 0.1,
                    "generated_token_count": 2,
                    "result_preview": "hello",
                },
            },
        )
    )

    clock.advance(1.0)
    path = metrics.write(status="success")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["duration_seconds"] == 8.0
    assert payload["model"]["moe_layer_indices"] == [0]
    assert payload["run_config"]["actual_total_tokens"] == 4
    assert payload["observer"]["per_layer"]["0"]["saliency_finite_count"] == 2
    assert payload["observer"]["per_layer"]["0"]["saliency_non_finite_count"] == 2
    assert payload["pruning"]["total_experts_removed"] == 1
    assert payload["save_reload"]["artifacts"]["total_bytes"] == 9
    assert payload["throughput"]["calibration_tokens_per_second"] == 2.0
    assert payload["throughput"]["generation_tokens_per_second"] == 20.0


def test_run_metrics_writes_failure_payload(tmp_path):
    clock = FakeClock()
    metrics = RunMetrics(tmp_path, clock=clock)
    metrics.sample_memory("failure")
    clock.advance(3.0)

    error = RuntimeError("boom")
    path = metrics.write(
        status="failed",
        failure=metrics.failure_payload("observe", error),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["failure"]["phase"] == "observe"
    assert payload["failure"]["type"] == "RuntimeError"
    assert payload["failure"]["message"] == "boom"


def test_record_model_metadata_handles_adapter_none_and_dense_model(tmp_path):
    """record_model_metadata must default gracefully when no adapter is detected."""
    metrics = RunMetrics(tmp_path)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=object()), SimpleNamespace(mlp=object())]
        )
    )
    config = {"model_type": "qwen3_moe", "num_hidden_layers": 2}
    metrics.record_model_metadata(model, config)  # adapter defaults to None

    m = metrics.data["model"]
    assert m["adapter_name"] is None
    assert m["moe_layer_indices"] == []
    assert m["moe_layer_count"] == 0
    assert m["dense_layer_indices"] == []
    assert m["layer_count"] == 2


def test_record_calibration_handles_empty_sequence_list(tmp_path):
    """Empty calibration list must record zeros, not crash on min/max of empty."""
    metrics = RunMetrics(tmp_path)
    metrics.record_calibration([])

    rc = metrics.data["run_config"]
    assert rc["actual_sample_count"] == 0
    assert rc["actual_total_tokens"] == 0
    assert rc["actual_min_tokens"] == 0
    assert rc["actual_max_tokens"] == 0
    assert rc["actual_mean_tokens"] == 0.0
    assert rc["actual_token_counts"] == []


def test_record_calibration_sums_varying_sequence_lengths(tmp_path):
    metrics = RunMetrics(tmp_path)
    metrics.record_calibration([
        {"input_ids": np.array([1, 2, 3])},
        {"input_ids": [4]},
        {"input_ids": np.array([5, 6])},
    ])

    rc = metrics.data["run_config"]
    assert rc["actual_sample_count"] == 3
    assert rc["actual_total_tokens"] == 6
    assert rc["actual_min_tokens"] == 1
    assert rc["actual_max_tokens"] == 3
    assert rc["actual_token_counts"] == [3, 1, 2]


def test_record_observer_counts_non_finite_saliency(tmp_path):
    metrics = RunMetrics(tmp_path)
    observer_data = {
        0: {"total_tokens": 2, "expert_frequency": [1, 1], "reap": [0.5, 2.0]},
        1: {"total_tokens": 3, "expert_frequency": [1, 0], "reap": [float("nan"), float("inf")]},
    }
    metrics.record_observer(observer_data, "reap")

    obs = metrics.data["observer"]
    assert obs["observed_moe_layer_count"] == 2
    assert obs["observed_moe_layer_indices"] == [0, 1]
    assert obs["total_input_tokens"] == 5
    assert obs["per_layer"]["0"]["saliency_non_finite_count"] == 0
    assert obs["per_layer"]["0"]["saliency_finite_count"] == 2
    assert obs["per_layer"]["1"]["saliency_non_finite_count"] == 2
    assert obs["per_layer"]["1"]["saliency_finite_count"] == 0


def test_record_observer_empty_observer_data(tmp_path):
    metrics = RunMetrics(tmp_path)
    metrics.record_observer({}, "reap")

    obs = metrics.data["observer"]
    assert obs["observed_moe_layer_count"] == 0
    assert obs["observed_moe_layer_indices"] == []
    assert obs["total_input_tokens"] == 0
    assert obs["per_layer"] == {}


def test_record_pruning_handles_none_keep_by_layer(tmp_path):
    metrics = RunMetrics(tmp_path)
    metrics.record_pruning(
        None,
        config_before={"num_experts": 4, "num_experts_per_tok": 2},
        config_after={"num_experts": 4, "num_experts_per_tok": 2},
        observer_data={},
    )

    p = metrics.data["pruning"]
    assert p["total_experts_removed"] == 0
    assert p["layer_count"] == 0
    assert p["per_layer"] == {}
    assert p["top_k_was_clamped"] is False


def test_record_pruning_records_zero_removed_noop(tmp_path):
    metrics = RunMetrics(tmp_path)
    observer_data = {0: {"expert_frequency": np.array([1, 2, 3, 4])}}
    metrics.record_pruning(
        {0: [0, 1, 2, 3]},  # keep all four experts
        config_before={"num_experts": 4, "num_experts_per_tok": 2},
        config_after={"num_experts": 4, "num_experts_per_tok": 2},
        observer_data=observer_data,
    )

    p = metrics.data["pruning"]
    assert p["total_experts_removed"] == 0
    assert p["per_layer"]["0"]["retained_expert_count"] == 4
    assert p["per_layer"]["0"]["removed_expert_count"] == 0


def test_derive_throughput_returns_none_when_no_phase_data(tmp_path):
    """No recorded phases, save_reload, or smoke must not divide by zero."""
    metrics = RunMetrics(tmp_path)
    metrics._derive_throughput()

    tp = metrics.data["throughput"]
    assert tp["calibration_tokens_per_second"] is None  # calibration_seconds == 0
    assert tp["observer_layers_per_second"] is None   # observe_seconds == 0
    assert tp["pruning_experts_per_second"] is None   # prune_seconds == 0
    assert tp["save_mb_per_second"] is None            # save_seconds missing
    assert tp["reload_mb_per_second"] is None           # reload_seconds missing
    assert tp["generation_tokens_per_second"] is None  # no smoke elapsed


def test_record_save_reload_falls_back_to_artifact_summary(tmp_path):
    """When result.metrics is None, record_save_reload uses artifact_summary."""
    metrics = RunMetrics(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    result = SimpleNamespace(
        output_dir=tmp_path,
        reloaded_config={"num_experts": 2},
        expected_expert_count=2,
        smoke_result=None,
        metrics=None,
    )
    metrics.record_save_reload(result, adapter=None)

    sr = metrics.data["save_reload"]
    assert sr["expected_expert_count"] == 2
    assert sr["reloaded_config_expert_count"] == 2
    assert sr["reloaded_adapter_moe_layer_count"] == 0
    assert sr["artifacts"]["total_bytes"] > 0
    assert metrics.data["smoke"]["enabled"] is False


def test_failure_payload_records_exception_type_and_message(tmp_path):
    metrics = RunMetrics(tmp_path, clock=FakeClock())
    payload = metrics.failure_payload("prune", ValueError("bad ratio"))

    assert payload["phase"] == "prune"
    assert payload["type"] == "ValueError"
    assert payload["message"] == "bad ratio"
    assert payload["elapsed_seconds_before_failure"] == 0.0  # FakeClock frozen at 0.0
