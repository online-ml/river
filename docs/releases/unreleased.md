# Unreleased

- Add `ppc64le` architecture to Linux wheel builds.
- Publish `abi3` (stable-ABI, `cp311-abi3`) wheels alongside the per-version native wheels. pip/uv keep using the faster native wheel whenever one matches your interpreter; the `abi3` wheel is a fallback so River still installs as a wheel on CPython versions that don't yet have native builds (e.g. a brand-new release).
- Moved `altair` from the runtime dependencies to the `docs`/`dev` groups; it was only used to draw plots in the docs, so installing River no longer pulls it in.
- Publish Pyodide/WebAssembly wheels (CPython 3.13 and 3.14) so River can run in the browser, e.g. via [JupyterLite](https://jupyterlite.readthedocs.io/) or `micropip`. The minimum `numpy` and `scipy` versions were lowered to `2.2.5` and `1.14.1` to match Pyodide.
- Display `compose.TransformerUnion` elements vertically in HTML representations.

## base

- `base.Transformer` and `base.SupervisedTransformer` are now properly abstract: their `transform_one` abstract method is registered with `abc`, so a subclass that forgets to implement it raises `TypeError` at instantiation instead of failing later. This also restores estimator-check coverage for all concrete transformers, which were unintentionally excluded from the automated test suite.

## anomaly

- Added `anomaly.LODA`, an online implementation of Pevný's *Lightweight on-line detector of anomalies*. It maintains an ensemble of one-dimensional `sketch.Histogram`s over sparse random projections and scores samples by their average negative log-likelihood.

## compose

- The mini-batch methods of `compose.Pipeline`, `compose.TransformerUnion`, `compose.Select`, and `compose.TransformerProduct` now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only, so a whole pipeline can be mini-batched on a non-pandas backend. The input backend is preserved on output, including the pandas index, and `pandas` is no longer required unless the input is a pandas frame. `TransformerProduct` keeps the pandas `Sparse[uint8]` fast path when crossing one-hot encoded features.
- `compose.Pipeline` now forwards extra keyword arguments (such as the timestamp `t` used by `utils.TimeRolling`, or a sample weight `w`) to each step whose method declares them, and drops them for steps that don't. This makes `feature_extraction.Agg`/`TargetAgg` backed by `utils.TimeRolling` work inside a pipeline via `model.learn_one(x, y, t=t)`. Routing applies to `learn_one` and to the predict-time methods (`predict_one`, `predict_proba_one`, `score_one`, `transform_one`), so it also works under `compose.learn_during_predict` where unsupervised steps learn during `predict_one(x, t=t)`. Fixes [#1600](https://github.com/online-ml/river/issues/1600). The accepted arguments are determined once when the pipeline plan is built, so pipelines with no extra arguments keep their previous speed.

## covariance

- Added weighted sample support to `EmpiricalCovariance.update` and `EmpiricalCovariance.revert` by accepting an optional `w` parameter and propagating it to the underlying `stats.Cov` and `stats.Var` statistics.
- Sped up `EmpiricalCovariance.update`/`revert` (~40% faster at 30 features) by caching the sorted feature list and pair iteration in the hot path. No semantic change.
- Restructured `EmpiricalPrecision` around NumPy-backed dense state, removing the per-update dict ↔ numpy marshalling. ~7× faster on 2000 × 20 sample streams.
- Fixed an `EmpiricalPrecision` asymmetry where features introduced at different times left the stored matrix skewed (e.g. `prec[a, b]` ≠ `prec[b, a]`).

## datasets

- Added `datasets.CriteoAds`, a 100,000-row sample of the Criteo Display Advertising Challenge (binary click prediction with 13 integer and 26 high-cardinality categorical features). A natural fit for one-hot models such as `linear_model.AdPredictor`.
- Added `datasets.Shuttle`, the UCI Statlog (Shuttle) dataset cast as a binary anomaly-detection task following the ODDS benchmark (49,097 observations, 9 numerical features, ~7% anomalies). Ships bundled with River.

## facto

- Sped up `learn_one` for all factorization-machine models by vectorizing the per-factor latent updates with NumPy instead of looping in Python. On MovieLens 100K: ~1.4× faster for `FFMRegressor`/`FFMClassifier`, ~1.8× for `FwFMRegressor`/`FwFMClassifier` and `HOFMRegressor`/`HOFMClassifier`. Outputs are unchanged.
- The factorization-machine models are now covered by the automated estimator checks (`utils.check_estimator`).

## feature_extraction

- Added proper mini-batch support to `feature_extraction.TFIDF`: `learn_many` now updates document frequencies, and `transform_many` returns TF-IDF weights. Both `feature_extraction.BagOfWords.transform_many` and `TFIDF.transform_many` now accept any narwhals-supported dataframe backend (pandas, polars, pyarrow, ...), as either a series of documents or a dataframe with the `on` parameter, and return the same backend (a sparse dataframe for pandas).

## linear_model

- Added `linear_model.AdPredictor`, the Bayesian online probit-regression classifier Microsoft used for click-through-rate prediction in Bing's sponsored search (Graepel et al., 2010). It keeps a Gaussian belief over each feature weight and yields well-calibrated probabilities.
- Restructured `BayesianLinearRegression` around NumPy-backed storage. ~11× faster `learn_one` at 20 features, ~24× at 50 features. Speeds up `bandit.LinUCB` too.
- `BayesianLinearRegression` now handles features arriving and disappearing after training begins (it passes `check_emerging_features` and `check_shuffle_features_no_impact`, previously skipped).
- Fixed `BayesianLinearRegression` coefficients diverging to `inf`/`nan` under emerging/disappearing features; `learn_one` now updates the full state with a zero-padded `x`. Behavior change: features absent from `x` are treated as observed 0s (matching the other linear models) rather than skipped — identical to before when every call sees the same features.
- Sped up the `LinearRegression`/`LogisticRegression.learn_many` mini-batch gradient (~2-3×) by contracting the sample axis inside the `np.einsum`. No semantic change.
- Sped up `learn_one` for the linear models (`LinearRegression`, `LogisticRegression`, `Perceptron`, ...): updates now scale with the number of active features instead of the total number of features ever seen. Outputs are unchanged.
- Stabilised `BayesianLinearRegression` across BLAS implementations and sped it up (~10-20%) by accumulating an exact natural mean and recovering the posterior mean lazily, instead of propagating it through compounding rank-1 updates (which drifted ~0.6% between macOS Accelerate and Linux OpenBLAS).
- `linear_model.LinearRegression` and `linear_model.LogisticRegression` mini-batch methods (`learn_many`, `predict_many`, `predict_proba_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only. The input backend is preserved on output, including the pandas index. These methods no longer require `pandas` to be installed.
- `linear_model.BayesianLinearRegression` is now a `MiniBatchRegressor`: it gained a `learn_many` method, equivalent to looping `learn_one` over the rows (exact without smoothing, and the matching closed-form geometric weighting with smoothing). Its `learn_many`/`predict_many` accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...), preserving the input backend and pandas index, and no longer require `pandas`.

## multioutput

- Added `multioutput.PerOutputClassifier`, the streaming equivalent of scikit-learn's `MultiOutputClassifier`. Trains one independent classifier per target output.
- Added `multioutput.PerOutputRegressor`, the streaming equivalent of scikit-learn's `MultiOutputRegressor`. Trains one independent regressor per target output, with no inter-output dependencies.

## naive_bayes

- Added mini-batch support to `GaussianNB` via `learn_many`, `predict_many`, and `predict_proba_many`.

## neural_net

- Removed the deprecated `river.neural_net` module (and its `MLPRegressor`), which had emitted a `DeprecationWarning` since 0.25.0. Use [`deep-river`](https://github.com/online-ml/deep-river) or a dedicated deep-learning library such as PyTorch for neural networks.

## optim

- Exposed `optim.Newton` (Online Newton Step), which was implemented but never exported, and fixed an initialisation bug (the inverse Hessian started at `eps * I` instead of `(1 / eps) * I`) that crippled learning. Reworked around NumPy-backed dense state.
- Fixed `optim.AdaBound` raising `TypeError` after being cloned (its base learning rate was captured as a scheduler instead of a number), which broke it inside `evaluate`, ensembles, model selection, and anywhere else estimators are cloned.
- Fixed `optim.NesterovMomentum` and `optim.FTRLProximal` raising when used to optimise estimators whose weights are stored as NumPy arrays, such as the factorization machines (`facto`).
- Added a test covering every optimizer against every estimator that accepts one, so optimizer/estimator incompatibilities are caught going forward.

## preprocessing

- Added a `window_size` parameter to `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler`, and `preprocessing.MaxAbsScaler`. When set, the scaler tracks its statistics over the last `window_size` observations instead of the whole stream.
- Added a `_from_state` classmethod to `preprocessing.MinMaxScaler`, `preprocessing.MaxAbsScaler`, and `preprocessing.StandardScaler` so a scaler can be warm-started from precomputed statistics without replaying past observations.
- `preprocessing.FeatureHasher` now hashes with MurmurHash3 in Rust, making it much faster. It gains an `alternate_sign` parameter (default `True`, matching scikit-learn) and returns a plain `dict`. Hashed feature indices differ from previous versions.
- `preprocessing.OneHotEncoder` mini-batch methods (`learn_many`, `transform_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only, preserving the input backend (including the pandas index) on output. The pandas path keeps returning `Sparse[uint8]` columns; other backends return dense integer columns, as they have no sparse-array equivalent. `transform_many` only requires `pandas` when the input is a pandas frame.
- `preprocessing.OrdinalEncoder` mini-batch methods (`learn_many`, `predict_many`, `predict_proba_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only. The input backend is preserved on output, including the pandas index. These methods no longer require `pandas` to be installed.
- `preprocessing.StandardScaler` mini-batch methods (`learn_many`, `transform_many`) now accept and return any [narwhals](https://github.com/narwhals-dev/narwhals)-supported eager backend (pandas, polars, pyarrow, ...) instead of being pandas-only, preserving the input backend (including the pandas index) on output. Classic numpy-backed pandas keeps the historical fast path (and its float dtype, e.g. `float32`); other backends are scaled through a `float64` path. `transform_many` only requires `pandas` when the input is a pandas frame. Outputs are unchanged on the pandas path.

## proba

- Added weighted sample support to `MultivariateGaussian.update` and `MultivariateGaussian.revert` by accepting an optional `w` parameter and propagating it to the underlying `EmpiricalCovariance` instance.

## reco

- Corrected the type annotations of the weight/latent `defaultdict`s in `BiasedMF`, `Baseline`, and `FunkMF`, and dropped the bespoke `reco.base.ID` alias in favour of `typing.Hashable`. Typing-only; no behavioral change.

## utils

- Added `utils.math.norm_cdf` and `utils.math.norm_pdf`, the CDF and PDF of the standard normal distribution (used by `linear_model.AdPredictor`).
- `utils.Rolling` and `utils.TimeRolling` now accept a class plus constructor keyword arguments, e.g. `utils.Rolling(stats.Mean, window_size=3)`. This avoids silently sharing state when they are used as `collections.defaultdict` factories. Passing a pre-built instance still works but is deprecated and will be removed in a future release.

## sketch

- Sped up `sketch.Histogram.update` by roughly 2× on typical data by operating on the underlying list directly and inlining the bin search, instead of going through `collections.UserList`. Outputs are unchanged.
- Fixed `sketch.Histogram.__add__`: merging two histograms now conserves the total count (point bins were previously double-counted) and sets `n` on the result, so `cdf` no longer raises on merged histograms. Merging with an empty histogram also works now.
- `sketch.Histogram.cdf` and `iter_cdf` now return `0.0` on an empty histogram instead of raising.

## rules

- Fixed `RecursionError` in `AMRules` on long streams: the `EBSTSplitter`, `TEBSTSplitter`, and `ExhaustiveSplitter` now traverse and deep-copy their search trees iteratively, so deeply-skewed trees no longer blow Python's recursion limit.
- Fixed an `AMRules` memory leak where `HoeffdingRule.expand` appended a redundant `NumericLiteral` when a new split shared a feature and direction with an existing literal without tightening the threshold.

## stats

- Added `stats.ChiSquared`, a streaming Chi-squared statistic between two categorical variables. Wrap it with `utils.Rolling` for a rolling version.

## stream

- Added `stream.iter_frame`, a dataframe-agnostic row iterator powered by [Narwhals](https://narwhals-dev.github.io/narwhals/) that works with any eager dataframe (pandas, polars, PyArrow, Modin, cuDF, ...).
- Deprecated `stream.iter_pandas` and `stream.iter_polars` in favour of `stream.iter_frame`. They now emit a `DeprecationWarning` and will be removed in a future release.
