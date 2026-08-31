# Unreleased

## stream

- `stream.Cache` now writes a pass to a temporary file and renames it into place once the stream is exhausted. An interrupted first pass (a `break`, an exception, an abandoned generator) used to leave a truncated file behind, which every later pass then read back as if it were the whole dataset.
- `stream.iter_csv` no longer yields an empty `x` for a blank line in the middle of a file. `csv.DictReader` skips those, and `stream.iter_arff` already did.
- `stream.iter_csv` now closes the file it opened and restores `csv.field_size_limit` even when the stream is not exhausted, e.g. when the caller breaks out of the loop. Only the file `iter_csv` opened itself is closed; a buffer passed in by the caller is still left open.
- `stream.iter_sql` now closes the result it iterates, so the underlying cursor is released once the stream is exhausted or abandoned.
- `stream.cache`, `stream.iter_csv`, and `stream.iter_sql` are now clean under strict mypy. `sqlalchemy` is type-checked rather than ignored, so the `query` and `conn` arguments of `stream.iter_sql` are checked against the SQLAlchemy 2.0 types.
- Enabled strict typing for `stream.iter_array`. It is now overloaded on its inputs: an `X` of texts is typed to yield `str` rows, and a plain sequence `y` propagates its own target type, while numpy arrays keep yielding `Any`. Typing-only, apart from the fixes below.
- Fixed `stream.iter_array` for plain Python inputs, which are documented as supported: `shuffle=True` no longer raises a `TypeError` on lists, and a multi-output `y` given as a list of lists no longer raises an `AttributeError`.
- `stream.iter_array` now yields an empty stream when `X` is empty, instead of raising an `IndexError`, and comes out slightly faster on every input shape.
- Added dedicated tests for `stream.iter_array`, covering every supported input container (numpy, list, tuple), shuffling, multi-output targets, text arrays, and the static types its overloads advertise.
