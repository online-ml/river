# Unreleased

## base

- The `base.BinaryClassifier` and `base.MultiClassifier` have been merge into `base.Classifier`. The `'binary_only'` tag is now used to indicate whether or not a classifier support multi-class classification or not.

## compose

- Fixed some bugs related to mini-batching in `compose.Pipeline`.

## datasets

- Added `datasets.SolarFlare`, which is a small multi-output regression dataset.

## decomposition

- `decomposition.LDA` now takes as input word counts instead of raw text.

## expert

- Created this new module, which will regroup methods that perform expert learning, which boils down to managing multiple models.
- Moved `ensemble.StackingBinaryClassifier` to `expert.StackingClassifier`.
- Moved `model_selection.SuccessiveHalvingClassifier` to `expert.SuccessiveHalvingClassifier`.
- Moved `model_selection.SuccessiveHalvingRegressor` to `expert.SuccessiveHalvingRegressor`.
- Moved `ensemble.HedgeRegressor` to `ensemble.EWARegressor`.

## evaluate

- Created this new module, which will contains methods for evaluating models.

## feature_extraction

- Moved `preprocessing.PolynomialExtender` to `feature_extraction.PolynomialExtender`.
- Moved `preprocessing.RBFSampler` to `feature_extraction.RBFSampler`.

## linear_model

- Added `linear_model.Perceptron`, which is implemented as a special case of logistic regression.

## model_selection

- Deleted this module.

## multiclass

- `multiclass.OneVsRestClassifier` now supports mini-batching.

## optim

- Removed `optim.MiniBatcher`.
- Implemented `optim.Averager`, which allows doing averaged stochastic gradient descent.
- Removed `optim.Perceptron`.

## utils

- Moved `model_selection.expand_param_grid` to `utils.expand_param_grid`.

## compat

- Added `compat.pytorch.PyTorch2RiverClassifier`
- Refactored `compat.pytorch.PyTorch2RiverRegressor`
