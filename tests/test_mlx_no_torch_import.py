"""Import-safety tests for the MLX backend package boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_SKIP_OPTIONAL_MLX_MISSING = 86


def _last_json_line(stdout: str) -> dict[str, object]:
    """Return the JSON payload printed by the subprocess helper."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        return json.loads(line)
    return {}


def test_mlx_backend_import_does_not_import_torch_or_vllm() -> None:
    """Importing reap.backends.mlx must not pull in Torch/vLLM stacks.

    This runs in a subprocess so any modules imported by the surrounding pytest
    session cannot contaminate the sys.modules checks. The current skeleton is
    importable without MLX installed; if future MLX APIs make an optional MLX
    dependency unavoidable during package import, report that as a clean skip.
    """
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_src)

    code = f"""
import importlib
import json
import sys

BLOCKED_MODULES = {[
    "reap.main",
    "reap.prune",
    "reap.layerwise_prune",
    "reap.eval",
    "reap.observer",
    "reap.layerwise_observer",
    "reap.data",
]!r}
BLOCKED_TOP_LEVELS = {[
    "torch",
    "vllm",
]!r}
OPTIONAL_MLX_MODULES = ("mlx", "mlx_lm")
SKIP_OPTIONAL_MLX_MISSING = {_SKIP_OPTIONAL_MLX_MISSING!r}

try:
    importlib.import_module("reap.backends.mlx")
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if any(missing == name or missing.startswith(f"{{name}}.") for name in OPTIONAL_MLX_MODULES):
        print(json.dumps({{"skip": f"optional MLX dependency missing: {{missing}}"}}))
        raise SystemExit(SKIP_OPTIONAL_MLX_MISSING)
    raise

loaded_top_levels = sorted(
    name
    for name in BLOCKED_TOP_LEVELS
    if name in sys.modules or any(module.startswith(f"{{name}}.") for module in sys.modules)
)
loaded_reap_modules = sorted(name for name in BLOCKED_MODULES if name in sys.modules)
loaded = loaded_top_levels + loaded_reap_modules
if loaded:
    print(json.dumps({{"loaded": loaded}}))
    raise SystemExit(1)

print(json.dumps({{"ok": True}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )

    if result.returncode == _SKIP_OPTIONAL_MLX_MISSING:
        payload = _last_json_line(result.stdout)
        pytest.skip(str(payload.get("skip", "optional MLX dependency missing")))

    assert result.returncode == 0, (
        "importing reap.backends.mlx in a clean subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    payload = _last_json_line(result.stdout)
    assert payload == {"ok": True}
