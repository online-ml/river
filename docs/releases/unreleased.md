# Unreleased

## bandit

- Added `bandit.BayesUCB`.
- Added `bandit.evaluate_offline`, for evaluating bandits on historical (logged) data.

## compat

- The `predict_many` method scikit-learn models wrapped with `compat.convert_sklearn_to_river` raised an exception if the model had been fitted on any data yet. Instead, default predictions will be produced, which is consistent with the rest of River.

## neighbors

- Add `neighbors.SWINN` to power-up approximate nearest neighbor search. SWINN uses graphs to speed up nearest neighbor search in large sliding windows of data.

## compose

- `compose.TransformerProduct` will now preserve the density of sparse columns.
- Added a `transform_many` method to `compose.FuncTransformer`, allowing it to be used in mini-batch pipelines.

## preprocessing

- Rename `sparse` parameter to `drop_zeros` in `preprocessing.OneHotEncoder`.
- The `transform_many` method of `preprocessing.OneHotEncoder` will now return a sparse dataframe, rather than a dense one, which will consume much less memory.

## proba

- Added a `cdf` method to `proba.Beta`.
