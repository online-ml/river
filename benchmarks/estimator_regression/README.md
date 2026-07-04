# Estimator regression CI

This suite runs public River ML algorithms on deterministic workloads, writes a
YAML metrics artifact, and compares a PR branch against its base branch.
## Local usage

```sh
# Run every scenario on the current tree:
make estimator-regression

# Focus on a scenario, harness, or workload:
make estimator-regression K=linear_model
make estimator-regression K=LogisticRegression

# Compare a locally generated base and head:
uv run python -m benchmarks.estimator_regression.cli run --output metrics.head.yml
uv run python -m benchmarks.estimator_regression.cli compare \
  --base metrics.base.yml --head metrics.head.yml --output report.md

# Inspect discovery and coverage:
make estimator-regression-discover
make estimator-regression-audit
```

CI is the source of truth for base/head comparison: the workflow checks out both
branches, runs the suite on each, and posts a Markdown summary as a PR comment.
`PYTHONHASHSEED=0` is required for reproducible runs.

The suite is a development tool and expects the same dependency set as CI,
including extras such as `compat` for the sklearn-backed scenarios.

## Coverage model

Every concrete public `base.Estimator` subclass must appear in `SCENARIOS` or
`EXCLUSIONS`. The `audit` subcommand fails CI if a discovered estimator is
missing, if an entry can no longer be imported, or if an exclusion lacks a
reason.

The suite covers ML algorithms only: classifiers, regressors, clusterers,
anomaly detectors, forecasters, recommenders, and wrapper families.
Transformers, feature extractors, preprocessors, composition helpers, and search
structures are excluded as "not an ML algorithm."

## Determinism contract

Every workload is materialized up front from a seeded RNG and every stochastic
estimator is constructed with `seed=42`. Run artifacts do not include wall-clock
timestamps, so repeated runs under `PYTHONHASHSEED=0` should be stable.

```sh
PYTHONHASHSEED=0 uv run python -m benchmarks.estimator_regression.cli run --output a.yml
PYTHONHASHSEED=0 uv run python -m benchmarks.estimator_regression.cli run --output b.yml
diff a.yml b.yml
```

## Comparison policy

Scenario IDs are permanent. Once an ID exists on `main`, its workload, harness,
and sample count should not change silently; `compare` reports a schema mismatch
when they do.

Metric tolerances live in the `TOLERANCES` dict in `compare.py`. A metric
regresses when the signed movement toward "worse" exceeds
`max(absolute_tolerance, abs(base) * relative_tolerance)`.

Any scenario error on the head branch fails the comparison, even if the same
scenario also errors on the base branch.
