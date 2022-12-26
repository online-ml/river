# Unreleased

## base

- Fixed an issue where an estimator that has attribute a pipeline could not be cloned.

## conf

- Introduced this new module to perform conformal predictions.
- Added a `conf.Interval` dataclass to represent predictive intervals.
- Added `conf.RegressionJackknife`.

## datasets

- Removed unnecessary Numpy usage in the `synth` submodule.
- Changed `np.random.RandomState` to `np.random.default_rng` where necessary.

## linear_model

- Renamed `use_dist` to `with_dist` in `linear_model.BayesianLinearRegression`'s `predict_one` method.

## multiclass

- Added a `coding_method` method to `multiclass.OCC` to control how the codes are randomly generated.

## preprocessing

- Renamed `alpha` to `fading_factor` in `preprocessing.AdaptiveStandardScaler`.

## rules

- Renamed `alpha` to `fading_factor` in `rules.AMRules`.

## sketch

- Renamed `alpha` to `fading_factor` in `sketch.HeavyHitters`.

## stats

- Renamed `alpha` to `fading_factor` in `stats.Entropy`.
- Renamed `alpha` to `fading_factor` in `stats.EWMean`.
- Renamed `alpha` to `fading_factor` in `stats.EWVar`.

## utils

- Removed artifacts from the merger.
