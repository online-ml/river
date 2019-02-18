# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `PolynomialExtender` to the `preprocessing` module
- `FuncExtractor` to the `feature_extraction` module
- `Discarder` to the `preprocessing` module
- `FMRegressor` to `linear_model`
- `FuncTransformer` to `preprocessing`

## [0.0.2] - 2019-02-13

### Added

- Passive-aggressive models to the `linear_model` module
- `HedgeClassifier` to the `ensemble` module
- `RandomDiscarder` to the `feature_selection` module
- `NUnique`, `Min`, `Max`, `PeakToPeak`, `Kurtosis`, `Skew`, `Sum`, `EWMean` to the `stats` module
- `AbsoluteLoss`, `HingeLoss`, `EpsilonInsensitiveHingeLoss` to the `optim` module
- `sklearn` wrappers to the `compat` module
- `TargetEncoder` to the `feature_extraction` module
- `NumericImputer` to the `impute` module

### Changed

- Made sure the running statistics produce the same results as `pandas`'s `rolling` method
