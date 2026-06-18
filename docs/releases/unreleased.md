# Unreleased

* Add `ppc64le` architecture to Linux wheel builds. (@ChidiebereNjoku)

## preprocessing

- Added `window_size` parameter to `preprocessing.MinMaxScaler` and `preprocessing.MaxAbsScaler`. When set, the scaler tracks the min/max (or absolute max) over the last `window_size` observations via `stats.RollingMin` / `stats.RollingMax` / `stats.RollingAbsMax` instead of the running statistic over the entire stream.
- Added `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from offline-computed statistics or resumed from a checkpoint without replaying past observations.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.
