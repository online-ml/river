# Unreleased

River's mini-batch methods now support pandas v2. In particular, River conforms to pandas' new sparse API.

## anomaly

- Added `anomaly.LocalOutlierFactor`, which is an online version of the LOF algorithm for anomaly detection that matches the scikit-learn implementation.
  - Made `score_one` method of `anomaly.LocalOutlierFactor` stateless
  - Defined default score for uninitialized detector

## clustering

- Add fixes to `cluster.DBSTREAM` algorithm, including:
  - Addition of the `-` sign before the `fading_factor` in accordance with the algorithm 2 proposed by Hashler and Bolanos (2016) to allow clusters with low weights to be removed.
  - The new `micro_cluster` is added with the key derived from the maximum key of the existing micro clusters. If the set of micro clusters is still empty (`len = 0`), a new micro cluster is added with key 0.
  - `cluster_is_up_to_date` is set to `True` at the end of the `self._recluster()` function.
  - shared density graph update timestamps are initialized with the current timestamp value

## datasets

- Added `datasets.WebTraffic`, which is a dataset that counts the occurrences of events on a website. It is a multi-output regression dataset with two outputs.

## forest

- Simplify inner the structures of `forest.ARFClassifier` and `forest.ARFRegressor` by removing redundant class hierarchy. Simplify how concept drift logging can be accessed in individual trees and in the forest as a whole.

## covariance

- Added `_from_state` method to `covariance.EmpiricalCovariance` to warm start from previous knowledge.

## proba

- Added `_from_state` method to `proba.MultivariateGaussian` to warm start from previous knowledge.

## tree

- Fix a bug in `tree.splitter.NominalSplitterClassif` that generated a mismatch between the number of existing tree branches and the number of tracked branches.
