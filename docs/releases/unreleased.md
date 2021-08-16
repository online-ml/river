# Unreleased

## base

- The `predict_many` and `predict_proba_many` methods have been removed from `base.Classifier`. They're part of `base.MiniBatchClassifier`.
## ensemble

- Implemented `ensemble.VotingClassifier`.
## meta

- Renamed `meta.TransformedTargetRegressor` to `meta.TargetTransformRegressor`.
- Added `meta.TargetStandardScaler`.
