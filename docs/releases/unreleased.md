# Unreleased

* Add `ppc64le` architecture to Linux wheel builds. (@ChidiebereNjoku)

## preprocessing

- Added `window_size` parameter to `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler`, and `preprocessing.MaxAbsScaler`. When set, the scaler tracks its statistics over the last `window_size` observations instead of the entire stream.
- Added `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from offline-computed statistics or resumed from a checkpoint without replaying past observations.

## utils

- `utils.Rolling` and `utils.TimeRolling` now accept a class as their first argument and forward extra keyword arguments to its constructor, e.g. `utils.Rolling(stats.Mean, window_size=3)` or `utils.Rolling(stats.Var, window_size=3, ddof=0)`. This avoids a footgun when using these wrappers as `collections.defaultdict` factories, where the previous instance form silently shared state across keys. Passing a pre-built instance still works but now emits a `DeprecationWarning` and will be removed in a future release.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.
