# Unreleased

## naive_bayes

- Added `CategoricalNB`, a Naive Bayes classifier for categorical features that maintains per-class frequencies for every value of every feature, with additive (Laplace) smoothing and support for both online and mini-batch modes.

## linear_model

- `BayesianLinearRegression.predict_dist_one` now replaces `BayesianLinearRegression.predict_one(..., with_dist=True)` to better respect the `Regressor` interface while still being able to predict distributions.

## anomaly

- Fixed `anomaly.PredictiveAnomalyDetection.score_one` mutating the model: it used to update the dynamic threshold statistics, so scoring a point changed the detector and repeated scoring of the same point returned different values. The threshold is now maintained by `learn_one` instead, leaving `score_one` side-effect free. Scores over the usual score-then-learn loop are unchanged. A new estimator check, `checks.anomaly.check_score_one_does_not_mutate`, now guards every anomaly detector (supervised or not) against `score_one` side effects.

- Fixed `anomaly.PredictiveAnomalyDetection` crashing with a `ZeroDivisionError` when scoring before the predictive model had learnt anything. The default `preprocessing.MinMaxScaler` maps a first observation to `0 / 0`, making the prediction and the squared error `NaN`; scoring then divided by a zero threshold, and the `NaN` went on to poison the dynamic threshold statistics permanently. A non-finite squared error is now treated as carrying no information: it scores `0.0` and is kept out of the running statistics.

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
