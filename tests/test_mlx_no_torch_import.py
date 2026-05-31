"""Import-safety tests for the REAP package root."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_importing_reap_does_not_import_heavy_runtime_packages():
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
                        "forbidden import during REAP package import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        import reap

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during REAP package import: "
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
