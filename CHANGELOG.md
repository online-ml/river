# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `PolynomialExtender`, `Discarder`, `FuncTransformer` to `preprocessing`
- `FMRegressor` to `linear_model`
- `Accuracy`, `MAE`, `MSE`, `RMSE`, `RMSLE`, `SMAPE` to `metrics`
- `CategoricalImputer` to `impute`
- `stats`:
    - `Mode`
    - `Quantile`
    - `RollingQuantile`
    - `Entropy`
    - `RollingMin`
    - `RollingMax`
    - `RollingMode`
    - `RollingSum`
    - `RollingPeakToPeak`
- `wrap_sklearn` to `compat`
- `SVD` to `reco`
- `BoxCoxTransformRegressor`, `TargetModifierRegressor` to `compose`
- `iter_csv` to `stream`
- `fetch_restaurants` to `datasets`
- `GaussianNB` to `naive_bayes`
- `Multinomial` and `Normal` to `dist`

## [0.0.2] - 2019-02-13

### Added

- Passive-aggressive models to `linear_model`
- `HedgeClassifier` to `ensemble`
- `RandomDiscarder` to `feature_selection`
- `NUnique`, `Min`, `Max`, `PeakToPeak`, `Kurtosis`, `Skew`, `Sum`, `EWMean` to `stats`
- `AbsoluteLoss`, `HingeLoss`, `EpsilonInsensitiveHingeLoss` to `optim`
- `sklearn` wrappers to `compat`
- `TargetEncoder` to `feature_extraction`
- `NumericImputer` to `impute`

### Changed

- Made sure the running statistics produce the same results as `pandas`'s `rolling` method
