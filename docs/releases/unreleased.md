# Unreleased

- Moved all the public modules imports from `river/__init__.py` to `river/api.py` and removed unnecessary dependencies between modules enabling faster cherry-picked import times (~3x).

## base

- Introduced an `mutate` method to the `base.Base` class. This allows setting attributes in a controlled manner, which paves the way for online AutoML. See [the recipe](/recipes/cloning-and-mutating) for more information.

## compat

- Moved the PyTorch wrappers to river-extra.

## compose

- Moved `utils.pure_inference_mode` to `compose.pure_inference_mode` and `utils.warm_up_mode` to `compose.warm_up_mode`.
- Pipeline parts can now be accessed by integer positions as well as by name.

## datasets

- Imports `synth`, enabling `from river import datasets; datasets.synth`.

## metrics

- Removed dependency to `optim`.
- Removed `metrics.Rolling`, due to the addition of `utils.Rolling`.
- Removed `metrics.TimeRolling`, due to the addition of `utils.Rolling`.

## proba

- Removed `proba.Rolling`, due to the addition of `utils.Rolling`.
- Removed `proba.TimeRolling`, due to the addition of `utils.Rolling`.

## stats

- Removed `stats.RollingMean`, due to the addition of `utils.Rolling`.
- Removed `stats.RollingVar`, due to the addition of `utils.Rolling`.
## stream

- `stream.iter_array` now handles text data.

## time_series

- Added `time_series.HorizonAggMetric`.

## utils

- Removed dependencies to `anomaly` and `compose`.
- Added `utils.Rolling` and `utils.TimeRolling`, which are generic wrappers for computing over a window (of time).
- Use binary search to speed-up element removal in `utils.SortedWindow`.
