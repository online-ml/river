# Unreleased

## evaluate

- Refactored `evaluate.progressive_validation` to work with `base.AnomalyDetector`s.

## facto

- Added `debug_one` method to `BaseFM`.

## feature_extraction

- Make the `by` parameter in `feature_extraction.Agg` and `feature_extraction.TargetAgg` to be optional, allowing to calculate aggregates over the whole data.
- Removed `feature_extraction.Lagger` and `feature_extraction.TargetLagger`. Their functionality can be reproduced by combining `feature_extraction.Agg` and `stats.Shift`.
- `feature_extraction.Agg` and `feature_extraction.Target` now have a `state` property. It returns a `pandas.Series` representing the current aggregates values within each group.

## metrics

- `metrics.ROCAUC` works with `base.AnomalyDetectors`s.
