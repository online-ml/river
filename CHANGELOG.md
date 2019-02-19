# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `PolynomialExtender` to `preprocessing`
- `FuncExtractor` to `feature_extraction`
- `Discarder` to `preprocessing`
- `FMRegressor` to `linear_model`
- `FuncTransformer` to `preprocessing`
- `Accuracy`, `MAE`, `MSE`, `RMSE`, `RMSLE` to `metrics`

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
