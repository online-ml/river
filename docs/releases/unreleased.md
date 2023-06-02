# Unreleased

## bandit

- Added `bandit.BayesUCB`.

## compat

- The `predict_many` method scikit-learn models wrapped with `compat.convert_sklearn_to_river` raised an exception if the model had been fitted on any data yet. Instead, default predictions will be produced, which is consistent with the rest of River.

## neighbors

- Add `neighbors.SWINN` to power-up approximate nearest neighbor search. SWINN uses graphs to speed up nearest neighbor search in large sliding windows of data.
- Add classes `neighbors.ANNClassifier` and `neighbors.ANNRegressor` to perform approximate nearest neighbor search in classification and regression tasks.

## proba

- Added a `cdf` method to `proba.Beta`.
