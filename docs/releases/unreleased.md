# Unreleased

* Add `ppc64le` architecture to Linux wheel builds. (@ChidiebereNjoku)

## linear_model

- `linear_model.LinearRegression` and `linear_model.LogisticRegression` mini-batch methods (`learn_many`, `predict_many`, `predict_proba_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only. The input backend is preserved on output, including the pandas index. These methods no longer require `pandas` to be installed.

## utils

- Added `utils.dataframe`, a small set of narwhals boundary helpers (`into_frame`, `into_series`, `to_native_series`, `to_native_frame`) for writing backend-agnostic mini-batch methods.
