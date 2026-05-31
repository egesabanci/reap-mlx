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

from reap.backends.mlx.validation_metrics import RunMetrics


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
        from reap.backends.mlx.validation_metrics import RunMetrics

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
