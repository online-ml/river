# Unreleased

- Add `ppc64le` architecture to Linux wheel builds.
- Moved `altair` from the runtime dependencies to the `docs`/`dev` groups; it was only used to draw plots in the docs, so installing River no longer pulls it in.
- Publish Pyodide/WebAssembly wheels (CPython 3.13 and 3.14) so River can run in the browser, e.g. via [JupyterLite](https://jupyterlite.readthedocs.io/) or `micropip`. The minimum `numpy` and `scipy` versions were lowered to `2.2.5` and `1.14.1` to match Pyodide.

## covariance

- Sped up `EmpiricalCovariance.update`/`revert` (~40% faster at 30 features) by caching the sorted feature list and pair iteration in the hot path. No semantic change.
- Restructured `EmpiricalPrecision` around NumPy-backed dense state, removing the per-update dict ↔ numpy marshalling. ~7× faster on 2000 × 20 sample streams.
- Fixed an `EmpiricalPrecision` asymmetry where features introduced at different times left the stored matrix skewed (e.g. `prec[a, b]` ≠ `prec[b, a]`).

## datasets

- Added `datasets.CriteoAds`, a 100,000-row sample of the Criteo Display Advertising Challenge (binary click prediction with 13 integer and 26 high-cardinality categorical features). A natural fit for one-hot models such as `linear_model.AdPredictor`.

## linear_model

- Added `linear_model.AdPredictor`, the Bayesian online probit-regression classifier Microsoft used for click-through-rate prediction in Bing's sponsored search (Graepel et al., 2010). It keeps a Gaussian belief over each feature weight and yields well-calibrated probabilities.
- Restructured `BayesianLinearRegression` around NumPy-backed storage. ~11× faster `learn_one` at 20 features, ~24× at 50 features. Speeds up `bandit.LinUCB` too.
- `BayesianLinearRegression` now handles features arriving and disappearing after training begins (it passes `check_emerging_features` and `check_shuffle_features_no_impact`, previously skipped).
- Fixed `BayesianLinearRegression` coefficients diverging to `inf`/`nan` under emerging/disappearing features; `learn_one` now updates the full state with a zero-padded `x`. Behavior change: features absent from `x` are treated as observed 0s (matching the other linear models) rather than skipped — identical to before when every call sees the same features.
- Sped up the `LinearRegression`/`LogisticRegression.learn_many` mini-batch gradient (~2-3×) by contracting the sample axis inside the `np.einsum`. No semantic change.
- Sped up `learn_one` for the linear models (`LinearRegression`, `LogisticRegression`, `Perceptron`, ...): updates now scale with the number of active features instead of the total number of features ever seen. Outputs are unchanged.
- Stabilised `BayesianLinearRegression` across BLAS implementations and sped it up (~10-20%) by accumulating an exact natural mean and recovering the posterior mean lazily, instead of propagating it through compounding rank-1 updates (which drifted ~0.6% between macOS Accelerate and Linux OpenBLAS).
- `linear_model.LinearRegression` and `linear_model.LogisticRegression` mini-batch methods (`learn_many`, `predict_many`, `predict_proba_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only. The input backend is preserved on output, including the pandas index. These methods no longer require `pandas` to be installed.

## multioutput

- Added `multioutput.PerOutputClassifier`, the streaming equivalent of scikit-learn's `MultiOutputClassifier`. Trains one independent classifier per target output.
- Added `multioutput.PerOutputRegressor`, the streaming equivalent of scikit-learn's `MultiOutputRegressor`. Trains one independent regressor per target output, with no inter-output dependencies.

## naive_bayes

- Added mini-batch support to `GaussianNB` via `learn_many`, `predict_many`, and `predict_proba_many`.

## optim

- Exposed `optim.Newton` (Online Newton Step), which was implemented but never exported, and fixed an initialisation bug (the inverse Hessian started at `eps * I` instead of `(1 / eps) * I`) that crippled learning. Reworked around NumPy-backed dense state.
- Fixed `optim.AdaBound` raising `TypeError` after being cloned (its base learning rate was captured as a scheduler instead of a number), which broke it inside `evaluate`, ensembles, model selection, and anywhere else estimators are cloned.
- Fixed `optim.NesterovMomentum` and `optim.FTRLProximal` raising when used to optimise estimators whose weights are stored as NumPy arrays, such as the factorization machines (`facto`).
- Added a test covering every optimizer against every estimator that accepts one, so optimizer/estimator incompatibilities are caught going forward.

## preprocessing

- Added a `window_size` parameter to `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler`, and `preprocessing.MaxAbsScaler`. When set, the scaler tracks its statistics over the last `window_size` observations instead of the whole stream.
- Added a `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from precomputed statistics without replaying past observations.
- `preprocessing.FeatureHasher` now hashes with MurmurHash3 in Rust, making it much faster. It gains an `alternate_sign` parameter (default `True`, matching scikit-learn) and returns a plain `dict`. Hashed feature indices differ from previous versions.

## reco

- Corrected the type annotations of the weight/latent `defaultdict`s in `BiasedMF`, `Baseline`, and `FunkMF`, and dropped the bespoke `reco.base.ID` alias in favour of `typing.Hashable`. Typing-only; no behavioral change.

## utils

- Added `utils.math.norm_cdf` and `utils.math.norm_pdf`, the CDF and PDF of the standard normal distribution (used by `linear_model.AdPredictor`).
- `utils.Rolling` and `utils.TimeRolling` now accept a class plus constructor keyword arguments, e.g. `utils.Rolling(stats.Mean, window_size=3)`. This avoids silently sharing state when they are used as `collections.defaultdict` factories. Passing a pre-built instance still works but is deprecated and will be removed in a future release.

## rules

- Fixed `RecursionError` in `AMRules` on long streams: the `EBSTSplitter`, `TEBSTSplitter`, and `ExhaustiveSplitter` now traverse and deep-copy their search trees iteratively, so deeply-skewed trees no longer blow Python's recursion limit.
- Fixed an `AMRules` memory leak where `HoeffdingRule.expand` appended a redundant `NumericLiteral` when a new split shared a feature and direction with an existing literal without tightening the threshold.

## stats

- Added `stats.ChiSquared`, a streaming Chi-squared statistic between two categorical variables. Wrap it with `utils.Rolling` for a rolling version.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.
