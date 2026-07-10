"""Tests for the pipeline checkpointing module."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reap.checkpoint import load_checkpoint, write_checkpoint


def test_checkpoint_module_import_does_not_import_heavy_runtime_packages():
    """Importing reap.checkpoint must not pull in mlx/mlx-lm/datasets/torch/vllm."""
    code = textwrap.dedent(
        """
        import importlib.util, sys
        before = set(sys.modules)
        importlib.util.find_spec("reap.checkpoint")
        import reap.checkpoint  # noqa: F401
        after = set(sys.modules)
        heavy = {"mlx", "mlx_lm", "datasets", "torch", "vllm"}
        assert not (heavy & after), heavy & after
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_write_and_load_checkpoint_round_trips_keep_by_layer(tmp_path):
    keep_by_layer = {
        0: np.array([1, 3]),
        1: np.array([0, 2, 4]),
    }
    path = tmp_path / "reap-checkpoint.json"
    write_checkpoint(
        path,
        keep_by_layer=keep_by_layer,
        config_before_prune={"num_experts": 8, "top_k": 4},
        model_name="mlx-community/test",
        prune_method="reap",
        compression_ratio=0.5,
        adapter_name="qwen3_moe",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["model_name"] == "mlx-community/test"
    assert payload["keep_by_layer"] == {"0": [1, 3], "1": [0, 2, 4]}
    assert payload["config_before_prune"]["num_experts"] == 8

    loaded = load_checkpoint(path)
    assert loaded["keep_by_layer"] == {0: [1, 3], 1: [0, 2, 4]}
    assert loaded["adapter_name"] == "qwen3_moe"


def test_load_checkpoint_accepts_version_1(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_name": "m",
                "adapter_name": "qwen3_moe",
                "prune_method": "reap",
                "compression_ratio": 0.5,
                "config_before_prune": {},
                "keep_by_layer": {"0": [1, 2]},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_checkpoint(path)
    assert loaded["keep_by_layer"] == {0: [1, 2]}


def test_load_checkpoint_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 999, "keep_by_layer": {"0": [1]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_empty_keep_by_layer(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "keep_by_layer": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no keep_by_layer"):
        load_checkpoint(path)


def test_write_checkpoint_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "reap-checkpoint.json"
    write_checkpoint(
        path,
        keep_by_layer={0: np.array([1])},
        config_before_prune={"num_experts": 4},
        model_name="m",
        prune_method="reap",
        compression_ratio=0.5,
        adapter_name="qwen3_moe",
    )
    assert path.exists()
