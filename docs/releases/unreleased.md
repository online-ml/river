# Unreleased

## base

- Renamed `base.WrapperMixin` to `base.Wrapper`.
- Introduced `base.WrapperEnsemble`.

## compat

- Added `compat.PyTorch2RiverClassifier`
- Implemented median absolute deviation in `stats.MAD`.
- Refactored `compat.PyTorch2RiverRegressor`
- Fixed an issue where some statistics could not be printed if they had not seen any data yet.

## ensemble

- Bug fixes in `ensemble.SRPClassifier` and `ensemble.SRPRegressor`.
- Some estimators have been moved into the `ensemble` module.

## optim

- `optim.Adam` and `optim.RMSProp` now work with `utils.VectorDict`s as well as `numpy.ndarray`s.

## selection

- This new module replaces the `expert` module.
- Implemented `selection.GreedyExpertRegressor`.

## stats

- Moved `model_selection.expand_param_grid` to `utils.expand_param_grid`.
- The `stats.Mean` and `stats.Var` implementations have been made more numerically stable.

## utils

- Added `utils.poisson`.
