# Contributing

Thanks for considering a contribution to REAP MLX. This project is focused on
MLX-LM expert pruning on Apple Silicon.

## Workflow

- Open an issue for non-trivial bug fixes, model adapters, or behavioral
  changes before implementation.
- Use a branch name that describes the change, such as
  `fix/mlx-loader-compatibility` or `feat/lfm2-adapter-tests`.
- Open a pull request against `main`.
- Keep pull requests scoped and reviewable.
- Do not commit generated model artifacts, local calibration outputs, cache
  directories, or files under `artifacts/`.

## Pull Request Titles

Pull request titles must follow Conventional Commits:

```txt
feat(mlx): add new model adapter
fix: handle mlx-lm loader compatibility
docs: update calibration guide
test: cover save reload validation
```

Squash merges use the pull request title, so this keeps `main` history readable.

## Development

Install dependencies:

```bash
uv sync --group dev
```

Run the focused test suite:

```bash
uv run python -m pytest -q tests/test_mlx_*.py
```

Run basic repository checks:

```bash
uv lock --check
git diff --check
uv run python -m reap.entrypoint --help
```

## Model And Runtime Notes

- CI runs on macOS arm64 because MLX is the target runtime.
- Prefer small fixture-style unit tests for architecture adapters.
- Real model E2E runs are useful before merging risky runtime changes, but
  large outputs must remain local and ignored.

## Security

Do not report security issues in public issues. See [SECURITY.md](SECURITY.md).
