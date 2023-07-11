# Unreleased

Calling `learn_one` in a pipeline will now update each part of the pipeline in turn. Before the unsupervised parts of the pipeline were updated during `predict_one`. This is more intuitive for new users. The old behavior, which yields better results, can be restored by calling `learn_one` with the new `compose.learn_during_predict` context manager.

## compose

- Removed the `compose.warm_up_mode` context manager.
- Removed the `compose.pure_inference_mode` context manager.
- The last step of a pipeline will be correctly updated if it is unsupervised, which wasn't the case before.

## linear_model

- Added a `predict_many` method to `linear_model.BayesianLinearRegression`.

## forest

- Fixed issue with `forest.ARFClassifier` which couldn't be passed a `CrossEntropy` metric.
- Fixed a bug in `forest.AMFClassifier` which slightly improves predictive accurary.
- Added `forest.AMFRegressor`.

## preprocessing

- Added `preprocessing.OrdinalEncoder`, to map string features to integers.

## utils

- Added `utils.random.exponential` to retrieve random samples following an exponential distribution.
