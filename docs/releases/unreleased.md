# Unreleased

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