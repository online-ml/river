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

## CodSpeed

CodSpeed runs the small pytest benchmark suite in `benchmarks/codspeed/` and the Rust
criterion benchmarks in `benches/` on every pull request and every push to `main`. Both
jobs use CodSpeed's CPU simulation mode, which produces deterministic instruction-count
measurements and flamegraphs. Pull requests get a CodSpeed comment plus status checks for
the Python benchmarks, Rust benchmarks, and the aggregate performance analysis.

### Run CodSpeed benchmarks locally

```sh
make benchmark                      # all Python benchmarks, local walltime table
make benchmark K=logistic           # filter by pytest -k expression
make benchmark-rust                 # all Rust criterion benches
make benchmark-rust BENCH=stats_bench
```

Local Python runs are smoke tests, not the merge-gating measurement. Outside CodSpeed CI,
`pytest-codspeed` falls back to a normal walltime table, so results depend on your
machine. The PR comment is the source of truth for comparisons. On Linux, power users can
run simulation locally with the CodSpeed CLI:

```sh
curl -fsSL https://codspeed.io/install.sh | bash
codspeed auth login
codspeed run -m simulation -- uv run --no-sync pytest benchmarks/codspeed --codspeed -o addopts=""
```

### Add a CodSpeed benchmark

Copy this template into `benchmarks/codspeed/test_<module>.py` and keep the benchmark
name stable after it lands; renaming a benchmark loses its CodSpeed history.

```python
from __future__ import annotations

import pytest

from river import <module>

from workloads import binary_stream  # or regression_stream, scalar_series, ...

pytestmark = pytest.mark.benchmark(group="<module>")


def test_<estimator>_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = <module>.<Estimator>(seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
```

Use deterministic workloads only:

- Seed every generator and every stochastic estimator.
- Do not use network access or read the clock inside a benchmark.
- Materialize data in `workloads.py`, outside the measured callable.
- Treat cached workload lists as frozen; do not mutate shared data.
- Build a fresh estimator inside each measured learning callable.

### Reading CodSpeed results

The PR comment links to each benchmark, its comparison against `main`, and the
differential flamegraph. A red CodSpeed performance status means at least one benchmark
regressed beyond the configured threshold. The default threshold is 10%; noisy benchmarks
can get a per-benchmark threshold in CodSpeed instead of loosening the global one.

Intentional regressions should be explained in the PR description and acknowledged by an
admin in the CodSpeed UI. Do not bypass GitHub branch protection for performance
regressions; acknowledgement keeps the dashboard and branch status consistent.
