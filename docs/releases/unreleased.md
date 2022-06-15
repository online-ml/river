# Unreleased

- Moved all the public modules imports from `river/__init__.py` to `river/api.py` and removed unnecessary dependencies between modules enabling faster cherry-picked import times (≈3x).

## base

- Introduced an `mutate` method to the `base.Base` class. This allows setting attributes in a controlled manner, which paves the way for online AutoML. See [/recipes/cloning-and-mutating] for more information.

## compat

- Moved the PyTorch wrappers to river-extra.

## compose

- Moved `utils.pure_inference_mode` to `compose.pure_inference_mode` and `utils.warm_up_mode` to `compose.warm_up_mode`.
- Pipeline parts can now be accessed by integer positions as well as by name.

## datasets

- Imports `synth`, enabling `from river import datasets; datasets.synth`.

## drift

- Refactor the concept drift detectors to match the remaining of River's API. Warnings are only issued by detectors that support this feature.
- Drifts can be assessed via the property `drift_detected`. Warning signals can be acessed by the property `warning_detected`. The update method now returns `self`.
- Ensure all detectors reset their inner states automatically after a concept drift detection.

## metrics

- Removed dependency to `optim`.
- Removed `metrics.Rolling`, due to the addition of `utils.Rolling`.
- Removed `metrics.TimeRolling`, due to the addition of `utils.Rolling`.

## proba

- Removed `proba.Rolling`, due to the addition of `utils.Rolling`.
- Removed `proba.TimeRolling`, due to the addition of `utils.Rolling`.

## rule
- The default `splitter` was changed to `tree.splitter.TEBST` for memory and running time efficiency.

## stats

- Removed `stats.RollingMean`, due to the addition of `utils.Rolling`.
- Removed `stats.RollingVar`, due to the addition of `utils.Rolling`.

## stream

- `stream.iter_array` now handles text data.

## time_series

- Added `time_series.HorizonAggMetric`.

## tree

- Rename `split_confidence` and `tie_threshold` to `delta` and `tau`, respectively. This way, the parameters are not misleading and match what the research papers have used for decades.
- Refactor `HoeffdingAdaptiveTree{Classifier,Regressor}` to allow the usage of any drift detector. Also, expose the significance level of the test used to switch between subtrees as a user-defined parameter.
- Correct test used to switch between foreground and background subtrees in `HoeffdingAdaptiveTreeRegressor`. Due to the continuous and unbounded nature of the monitored errors, a z-test is now performed to decide which subtree to keep.
- The default `leaf_prediction` value was changed to `"adaptive"`, as this often results in the smallest errors in practice.
- The default `splitter` was changed to `tree.splitter.TEBST` for memory and running time efficiency.

## utils

- Removed dependencies to `anomaly` and `compose`.
- Added `utils.Rolling` and `utils.TimeRolling`, which are generic wrappers for computing over a window (of time).
- Use binary search to speed-up element removal in `utils.SortedWindow`.
