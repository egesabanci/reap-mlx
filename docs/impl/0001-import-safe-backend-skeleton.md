# 0001 - Import-Safe Backend Skeleton

## Issue Goal

Create the smallest MLX backend namespace needed for future work while keeping
all existing PyTorch/CUDA behavior untouched. Phase 0 is a skeleton only: it must
be importable without loading heavyweight runtime packages or existing eager
CUDA-oriented modules.

## Files Created

- `src/reap/backends/__init__.py` - import-light namespace for optional backend
  implementations.
- `src/reap/backends/mlx/__init__.py` - import-light MLX backend namespace with
  metadata only.
- `tests/test_mlx_no_torch_import.py` - subprocess import-safety regression
  test.
- `docs/impl/0001-import-safe-backend-skeleton.md` - implementation notes and
  contract for this phase.

## Import-Safety Contract

- `import reap.backends` and `import reap.backends.mlx` must succeed with only
  `PYTHONPATH=src`.
- Backend namespace imports must not import `torch`, `vllm`, `mlx`, `mlx_lm`, or
  existing eager CUDA/PyTorch implementation modules.
- Optional runtime dependencies belong in future concrete implementation modules
  or inside explicit functions, not package `__init__` files.
- Phase 0 does not expose router, observer, pruning, evaluation, or CLI APIs.

## Test Strategy

`tests/test_mlx_no_torch_import.py` launches a fresh Python subprocess with
`PYTHONPATH` set to `src`. Inside that subprocess it installs a temporary
`sys.meta_path` import blocker for the root packages `torch`, `vllm`, `mlx`, and
`mlx_lm`, imports `reap.backends.mlx`, and then checks `sys.modules` to ensure no
blocked package or submodule was loaded. It also asserts that known eager REAP
modules such as `reap.data`, `reap.eval`, `reap.main`, and `reap.prune` were not
loaded indirectly. Running in a subprocess avoids pytest plugins or the parent
test process polluting the import state.

## Safe Commands

- `PYTHONPATH=src pytest -q tests/test_mlx_no_torch_import.py`
- `git diff --check`
- `PYTHONPATH=src python3 -c "import reap.backends.mlx"`

## Deferred Work

- Backend registry or routing selection.
- MLX model adapters and mlx-lm integration.
- Router observation, pruning, merging, evaluation, and CLI support.
- Any changes to existing PyTorch/CUDA modules.
