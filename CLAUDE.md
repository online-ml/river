# CLAUDE.md

## What is River?

River is a Python library for online (streaming) machine learning. All estimators implement incremental `learn_one`/`predict_one` methods (or `learn_many`/`predict_many` for mini-batch).

## Common commands

```sh
# Install / sync dependencies (also builds Cython + Rust extensions)
uv sync

# Run tests (excludes datasets and slow markers by default, includes doctests)
uv run pytest

# Run a single test file
uv run pytest river/linear_model/test_glm.py

# Run tests in parallel
uv run pytest -n auto

# Run only web-dependent tests (datasets downloads)
uv run pytest -m datasets

# Lint and format (via prek hooks)
prek run --all-files

# Type checking
uv run mypy river

# Build and serve docs locally
uv sync --group docs
make livedoc
```

## Architecture

### Base classes (`river/base/`)

All estimators inherit from `base.Estimator` (which inherits from `base.Base`). Key interfaces:
- `Classifier` / `MiniBatchClassifier` — classification
- `Regressor` / `MiniBatchRegressor` — regression
- `Transformer` / `SupervisedTransformer` — feature transformation
- `Clusterer` — clustering
- `DriftDetector` / `BinaryDriftDetector` — concept drift detection
- `Ensemble` / `WrapperEnsemble` — ensemble methods
- `Wrapper` — wrapping other estimators

### Estimator conventions

- `learn_one(x, y)` / `predict_one(x)` is the core online learning interface
- `__init__` parameters need type hints; provide defaults or implement `_unit_test_params()`
- `_unit_test_skips()` returns check names to skip in automated testing
- `_supervised`, `_multiclass`, `_is_stochastic`, `_tags` are special class attributes
- Pipeline composition: `scaler | model` (uses `__or__`), `+` for parallel union

### Estimator testing framework (`river/checks/`, `river/test_estimators.py`)

- `utils.check_estimator(MyEstimator)` automatically discovers and runs validation checks (repr, cloning, pickling, feature robustness, etc.). New estimators must pass all checks.
- Pytest is configured with `--doctest-modules` — docstring examples are executed as tests.

### Extensions

- **Cython**: `.pyx` files throughout `river/` for performance-critical code
- **Rust**: `rust_src/lib.rs` via PyO3, exposed as `river.stats._rust_stats`

### Making changes

- When you're done, add a entry to `unreleased.md` if its relevant to end users.
- Performance matters: if you make a significant change, run a benchmark.
