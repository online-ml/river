# Unreleased

River's mini-batch methods now support pandas v2. In particular, River conforms to pandas' new sparse API.

## anomaly

- Added `anomaly.LocalOutlierFactor`, which is an online version of the LOF algorithm for anomaly detection that matches the scikit-learn implementation.

## forest

- Simplify inner the structures of `forest.ARFClassifier` and `forest.ARFRegressor` by removing redundant class hierarchy. Simplify how concept drift logging can be accessed in individual trees and in the forest as a whole.

## covariance

- Added `_from_state` method to `covariance.EmpiricalCovariance` to warm start from previous knowledge

## proba

- Added `_from_state` method to `proba.MultivariateGaussian` to warm start from previous knowledge