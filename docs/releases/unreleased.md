# Unreleased

## compose

- You can now use a `list` as a shorthand to build a `TransformerUnion`.
- Fixed a visualization issue when using a pipeline with multiple feature unions.

## ensemble

- Bug fixes in `SRPClassifier` and `SRPRegressor`.

## feature_extraction

- Implemented `feature_extraction.Lagger`.
- Implemented `feature_extraction.TargetLagger`.

## optim

- `optim.Adam` and `optim.RMSProp` now work with `utils.VectorDict`s as well as `numpy.ndarray`s.

## stats

- Fixed an issue where some statistics could not be printed if they had not seen any data yet.
- Implemented median absolute deviation in `stats.MAD`.

## time_series

- `time_series.Detrender` and `time_series.GroupDetrender` have been removed as they overlap with `meta.TargetStandardScaler`.
- Implemented a `time_series.evaluate` method, which performs progressive validation for time series scenarios.
- Implemented `time_series.HorizonMetric` class to evaluate the performance of a forecasting model at each time step along a horizon.
