# Unreleased

* Add `ppc64le` architecture to Linux wheel builds.

## covariance

- Sped up `EmpiricalCovariance.update`/`revert` by caching the sorted feature list and pair iteration and by removing the `__getitem__`/`matrix` indirection in the hot path. ~40% faster at 30 features, no semantic change (pairwise-deletion semantics preserved).
- Restructured `EmpiricalPrecision` to use NumPy-backed dense state indexed by a feature → integer map, eliminating the dict ↔ numpy marshalling on every `update`/`update_many`. ~7× faster on 2000 × 20 sample streams.
- Fixed a latent asymmetry in `EmpiricalPrecision` under emerging features: the per-feature `w` scaling left the stored matrix skewed (e.g. `prec[a, b]` ≠ `prec[b, a]`) when features were introduced at different times.

## linear_model

- Restructured `BayesianLinearRegression` to use the same NumPy-backed storage as `EmpiricalPrecision`. ~11× faster `learn_one` at 20 features, ~24× at 50 features. Speeds up `bandit.LinUCB` as a side effect.
- `BayesianLinearRegression` now passes `check_emerging_features` and `check_shuffle_features_no_impact` (the two checks previously skipped via `_unit_test_skips`). It now handles features arriving and disappearing after training begins.

## preprocessing

- Added `window_size` parameter to `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler`, and `preprocessing.MaxAbsScaler`. When set, the scaler tracks its statistics over the last `window_size` observations instead of the entire stream.
- Added `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from offline-computed statistics or resumed from a checkpoint without replaying past observations.

## utils

- `utils.Rolling` and `utils.TimeRolling` now accept a class as their first argument and forward extra keyword arguments to its constructor, e.g. `utils.Rolling(stats.Mean, window_size=3)` or `utils.Rolling(stats.Var, window_size=3, ddof=0)`. This avoids a footgun when using these wrappers as `collections.defaultdict` factories, where the previous instance form silently shared state across keys. Passing a pre-built instance still works but now emits a `DeprecationWarning` and will be removed in a future release.

## rules

- Fixed `RecursionError` in `AMRules` on long streams: `tree.splitter.EBSTSplitter` (and `TEBSTSplitter`) now traverses its binary search tree iteratively and the BST nodes carry a custom iterative `__deepcopy__`, so deeply-skewed trees no longer blow Python's recursion limit when rules are cloned during expansion. `tree.splitter.ExhaustiveSplitter` received the same treatment (iterative split-search, iterative node insertion, and iterative `__deepcopy__`).
- Fixed an `AMRules` memory leak where `HoeffdingRule.expand` appended a redundant `NumericLiteral` whenever a new split shared the feature and direction of an existing literal but did not tighten the threshold.

## stats

- Added `stats.ChiSquared`, a streaming Chi-squared statistic between two categorical variables. Wrap it with `utils.Rolling` for a rolling version.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.