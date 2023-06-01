# Unreleased

## bandit

- Added `bandit.BayesUCB`.

## compat

- The `predict_many` method scikit-learn models wrapped with `compat.convert_sklearn_to_river` raised an exception if the model had been fitted on any data yet. Instead, default predictions will be produced, which is consistent with the rest of River.

## proba

- Added a `cdf` method to `proba.Beta`.
