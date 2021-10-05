# Unreleased

## compose

- You can now use a `list` as a shorthand to build a `TransformerUnion`.

## ensemble

- Bug fixes in `SRPClassifier` and `SRPRegressor`.

## feature_extraction

- Implemented `feature_extraction.Lagger`.
- - Implemented `feature_extraction.TargetLagger`.

## optim

- `optim.Adam` and `optim.RMSProp` now work with `utils.VectorDict`s as well as `numpy.ndarray`s.

## stats

- Fixed an issue where some statistics could not be printed if they had not seen any data yet.
- Implemented median absolute deviation in `stats.MAD`.

## time_series

- `time_series.Detrender` and `time_series.GroupDetrender` have been removed as they overlap with `meta.TargetStandardScaler`.
