# Unreleased

River's mini-batch methods now support pandas v2. In particular, River conforms to pandas' new sparse API.

## anomaly

- Added `anomaly.LocalOutlierFactor`, which is an online version of the LOF algorithm for anomaly detection that matches the scikit-learn implementation.

## datasets

- Added `datasets.WebTraffic`, which is a dataset that counts the occurrences of events on a website. It is a multi-output regression dataset with two outputs.

## forest

- Simplify inner the structures of `forest.ARFClassifier` and `forest.ARFRegressor` by removing redundant class hierarchy. Simplify how concept drift logging can be accessed in individual trees and in the forest as a whole.

## reco

- Renamed `reco.base.Ranker` to `reco.base.Recommender`.
- Added a `sample(user, items)` method to `reco.base.Recommender`.
