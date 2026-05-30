"""Import-safety tests for the MLX backend namespace."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_importing_mlx_backend_does_not_import_heavy_runtime_packages():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    code = textwrap.dedent(
        """
        import sys

        BLOCKED_ROOTS = ("torch", "vllm", "mlx", "mlx_lm")
        EAGER_REAP_MODULES = (
            "reap.data",
            "reap.eval",
            "reap.layerwise_observer",
            "reap.layerwise_prune",
            "reap.main",
            "reap.model_util",
            "reap.models",
            "reap.observer",
            "reap.prune",
        )

        def is_blocked(fullname):
            return any(
                fullname == root or fullname.startswith(root + ".")
                for root in BLOCKED_ROOTS
            )

        class ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if is_blocked(fullname):
                    raise AssertionError(
                        "forbidden import during MLX backend import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        import reap.backends.mlx

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX backend import: "
                + ", ".join(forbidden_loaded)
            )

        eager_reap_loaded = sorted(
            name for name in sys.modules if name in EAGER_REAP_MODULES
        )
        if eager_reap_loaded:
            raise AssertionError(
                "eager REAP modules loaded during MLX backend import: "
                + ", ".join(eager_reap_loaded)
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
