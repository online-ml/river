# Unreleased

## stream

- `stream.Cache` now writes a pass to a temporary file and renames it into place once the stream is exhausted. An interrupted first pass (a `break`, an exception, an abandoned generator) used to leave a truncated file behind, which every later pass then read back as if it were the whole dataset.
- `stream.iter_csv` no longer yields an empty `x` for a blank line in the middle of a file. `csv.DictReader` skips those, and `stream.iter_arff` already did.
- `stream.iter_csv` now closes the file it opened and restores `csv.field_size_limit` even when the stream is not exhausted, e.g. when the caller breaks out of the loop. Only the file `iter_csv` opened itself is closed; a buffer passed in by the caller is still left open.
- `stream.iter_sql` now closes the result it iterates, so the underlying cursor is released once the stream is exhausted or abandoned.
- `stream.cache`, `stream.iter_csv`, and `stream.iter_sql` are now clean under strict mypy. `sqlalchemy` is type-checked rather than ignored, so the `query` and `conn` arguments of `stream.iter_sql` are checked against the SQLAlchemy 2.0 types.
