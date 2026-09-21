# Unreleased

## drift

- Added `drift.binary.WSTD`, an incremental Wilcoxon rank-sum two-window drift detector for binary error streams (Barros, Hidalgo & Cabral, 2018).
- The `river.drift` sub-package is now clean under strict mypy, and the `river.drift.*` entry was removed from the non-strict overrides in `pyproject.toml`. Public signatures and docstrings are unchanged.
- Fixed a latent bug in `drift.datasets.base.Dataset._annotations_aggregated`, which read a nonexistent `self._annotations` attribute and keyed the intersection branch on the integer `0` instead of the annotator key. The method has no callers today, so behavior is unchanged.
- `DriftRetrainingClassifier` now passes a `bool` instead of an `int` as the error indicator to its wrapped binary drift detector.

## naive_bayes

- Added `CategoricalNB`, a Naive Bayes classifier for categorical features that maintains per-class frequencies for every value of every feature, with additive (Laplace) smoothing and support for both online and mini-batch modes.

## linear_model

- `BayesianLinearRegression.predict_dist_one` now replaces `BayesianLinearRegression.predict_one(..., with_dist=True)` to better respect the `Regressor` interface while still being able to predict distributions.

## naive_bayes

- `MultinomialNB.learn_many`, `predict_many`, and `predict_proba_many` now accept any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only, preserving the input backend (including the pandas index) on output. A backend-agnostic `BaseNB.predict_many` was added so the argmax-over-probabilities logic is shared by all Naive Bayes variants.
## stream

- `stream.Cache` now writes a pass to a temporary file and renames it into place once the stream is exhausted. An interrupted first pass (a `break`, an exception, an abandoned generator) used to leave a truncated file behind, which every later pass then read back as if it were the whole dataset.
- `stream.iter_csv` no longer yields an empty `x` for a blank line in the middle of a file. `csv.DictReader` skips those, and `stream.iter_arff` already did.
- `stream.iter_csv` now closes the file it opened and restores `csv.field_size_limit` even when the stream is not exhausted, e.g. when the caller breaks out of the loop. Only the file `iter_csv` opened itself is closed; a buffer passed in by the caller is still left open.
- `stream.iter_sql` now closes the result it iterates, so the underlying cursor is released once the stream is exhausted or abandoned.
- `stream.cache`, `stream.iter_csv`, and `stream.iter_sql` are now clean under strict mypy. `sqlalchemy` is type-checked rather than ignored, so the `query` and `conn` arguments of `stream.iter_sql` are checked against the SQLAlchemy 2.0 types.

## preprocessing

- `preprocessing.Normalizer` now handles zero vectors without raising a `ZeroDivisionError`. A zero vector is returned unchanged instead.

## cluster

- Fixed the radius of `cluster.DenStream` micro-clusters, which used the norm of the vector of squared sums instead of the sum of its components and was 0 away from the origin. The docstring example now uses `epsilon=1.0`.
- `cluster.DenStream.predict_one` now expands a cluster past the direct neighbors of its first micro-cluster. The neighbors of a neighbor were only queued if they already had a label, so a chain of micro-clusters came out as several clusters.
- New `cluster.DenStream` micro-clusters no longer reuse the key of a deleted one, which overwrote an existing micro-cluster.
