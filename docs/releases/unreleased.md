# Unreleased

* Add `ppc64le` architecture to Linux wheel builds.
* Dropped `altair` from River's runtime dependencies. It was never imported by the package itself (it is only used to draw plots in the documentation notebooks), so it has been moved to the `docs`/`dev` dependency groups. Installing River no longer pulls in `altair` and its transitive dependencies.
* Publish Pyodide/WebAssembly wheels (CPython 3.13 and 3.14) so River can be installed in the browser, e.g. via [JupyterLite](https://jupyterlite.readthedocs.io/) or `micropip`. The minimum `numpy` and `scipy` versions were lowered to `2.2.5` and `1.14.1` to match the versions bundled with Pyodide.

## covariance

- Sped up `EmpiricalCovariance.update`/`revert` by caching the sorted feature list and pair iteration and by removing the `__getitem__`/`matrix` indirection in the hot path. ~40% faster at 30 features, no semantic change (pairwise-deletion semantics preserved).
- Restructured `EmpiricalPrecision` to use NumPy-backed dense state indexed by a feature → integer map, eliminating the dict ↔ numpy marshalling on every `update`/`update_many`. ~7× faster on 2000 × 20 sample streams.
- Fixed a latent asymmetry in `EmpiricalPrecision` under emerging features: the per-feature `w` scaling left the stored matrix skewed (e.g. `prec[a, b]` ≠ `prec[b, a]`) when features were introduced at different times.

## linear_model

- Restructured `BayesianLinearRegression` to use the same NumPy-backed storage as `EmpiricalPrecision`. ~11× faster `learn_one` at 20 features, ~24× at 50 features. Speeds up `bandit.LinUCB` as a side effect.
- `BayesianLinearRegression` now passes `check_emerging_features` and `check_shuffle_features_no_impact` (the two checks previously skipped via `_unit_test_skips`). It now handles features arriving and disappearing after training begins.
- Fixed `BayesianLinearRegression` coefficient blow-up under emerging/disappearing features. The previous submatrix-only update broke the `_ss_inv ≈ inv(_ss)` invariant once different feature subsets were touched across calls (the submatrix of an inverse is generally not the inverse of the submatrix), causing coefficients to diverge to `inf`/`nan` on `check_emerging_features`-style streams. `learn_one` now updates the full state with a zero-padded `x`. Behavior change: features absent from `learn_one`'s `x` are now treated as observed values of 0 (matching `LinearRegression` and the rest of the online-learning estimators), rather than being silently skipped. Identical to the previous behavior to floating-point roundoff when every call sees the same feature set.
- Sped up the mini-batch gradient in `LinearRegression`/`LogisticRegression.learn_many` by contracting the sample axis directly inside the `np.einsum` call instead of building the intermediate `(n, p)` matrix and averaging it afterwards. ~2-3× faster on that step, no semantic change.
- Stabilised `BayesianLinearRegression` across BLAS implementations and sped it up further. `learn_one` now accumulates an exact natural mean `_eta_arr = beta * sum_i(y_i * x_i)` alongside the existing Sherman-Morrison rank-1 update of `_ss_inv_arr`, and the posterior mean is recovered lazily as `_ss_inv_arr @ _eta_arr` (cached and invalidated on `learn_one`). Previously the posterior mean was propagated through `m_new = ss_inv @ (ss_old @ m_old + bx*y)`, which compounded BLAS rounding multiplicatively across rank-1 updates and caused a ~0.6% relative drift between macOS Accelerate and Linux OpenBLAS on `TrumpApproval` with imputation. The new path keeps `learn_one` ~20% faster than before and the full `predict`+`learn` cycle ~10% faster.

## optim

- Exposed `optim.Newton`, the Online Newton Step optimizer, which was previously implemented but never exported. Fixed a correctness bug whereby the inverse Hessian was initialized to `eps * I` instead of `(1 / eps) * I`, which crippled learning (the maintained inverse could only ever shrink from a tiny starting value). Reworked the internals to use NumPy-backed dense state and `utils.math.sherman_morrison` (the same BLAS rank-1 update used by `BayesianLinearRegression`) instead of a bespoke dict-based reimplementation.
## preprocessing

- Added `window_size` parameter to `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler`, and `preprocessing.MaxAbsScaler`. When set, the scaler tracks its statistics over the last `window_size` observations instead of the entire stream.
- Added `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from offline-computed statistics or resumed from a checkpoint without replaying past observations.

## reco

- Corrected the type annotations of the weight/latent `defaultdict`s in `BiasedMF`, `Baseline`, and `FunkMF`: their values are floats/arrays (not `Initializer` objects), and their keys are hashable IDs. Removed the bespoke `reco.base.ID` alias in favour of the built-in `typing.Hashable` (matching `base.typing.FeatureName`). Typing-only; no behavioral change.

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
