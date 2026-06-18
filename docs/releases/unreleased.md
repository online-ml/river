# Unreleased

* Add `ppc64le` architecture to Linux wheel builds.

## covariance

- Sped up `EmpiricalCovariance.update`/`revert` by caching the sorted feature list and pair iteration and by removing the `__getitem__`/`matrix` indirection in the hot path. ~40% faster at 30 features, no semantic change (pairwise-deletion semantics preserved).
- Restructured `EmpiricalPrecision` to use NumPy-backed dense state indexed by a feature → integer map, eliminating the dict ↔ numpy marshalling on every `update`/`update_many`. ~7× faster on 2000 × 20 sample streams.
- Fixed a latent asymmetry in `EmpiricalPrecision` under emerging features: the per-feature `w` scaling left the stored matrix skewed (e.g. `prec[a, b]` ≠ `prec[b, a]`) when features were introduced at different times.

## linear_model

- Restructured `BayesianLinearRegression` to use the same NumPy-backed storage as `EmpiricalPrecision`. ~11× faster `learn_one` at 20 features, ~24× at 50 features. Speeds up `bandit.LinUCB` as a side effect.
- `BayesianLinearRegression` now passes `check_emerging_features` and `check_shuffle_features_no_impact` (the two checks previously skipped via `_unit_test_skips`). It now handles features arriving and disappearing after training begins.
