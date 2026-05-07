# River Benchmarks

River benchmarks use [airspeed velocity (asv)](https://asv.readthedocs.io/) to track the timing performance of models across git history.

Each benchmark runs progressive validation (`evaluate.iter_progressive_val_score`) for a model on a dataset, measuring wall-clock time. Results are tracked per commit so that performance regressions can be detected automatically.

## Prerequisites

```bash
pip install asv virtualenv
```

## Running benchmarks locally

All commands should be run from the `benchmarks/` directory.

### Quick run (current Python, no environment setup)

```bash
asv run --python=same --quick --show-stderr
```

### Run a specific benchmark class

```bash
asv run --python=same --quick --bench BinaryClassification
asv run --python=same --quick --bench Regression
```

### Run a specific model/dataset combination

```bash
asv run --python=same --quick --bench "BinaryClassification.*Logistic.*Bananas"
```

### Compare two branches

```bash
asv continuous main my-feature-branch --bench BinaryClassification
```

This runs benchmarks on both branches and highlights significant timing changes.

### Find which commit caused a regression

```bash
asv find v0.22.0..main --bench "Regression.time_progressive_val_score\(Linear Regression, ChickWeights\)"
```

## Viewing results

```bash
asv publish
asv preview
```

This generates a static website and serves it locally. Open the printed URL in your browser.

## Adding a new model

1. Edit `benchmarks/common.py`
2. Add the model name to the appropriate `*_MODELS` list (e.g., `BINARY_CLF_MODELS`)
3. Add a factory lambda to the corresponding `_*_MODEL_REGISTRY` dict
4. Run `asv check` to verify discovery

## Adding a new track

1. Create `benchmarks/bench_<track_name>.py` following the pattern in existing files
2. Add the model/dataset lists and registries to `benchmarks/common.py`
3. Run `asv check` to verify discovery

## CI

Benchmarks run automatically on push to `main` via GitHub Actions. Results are stored on the `asv` branch and published to GitHub Pages.
