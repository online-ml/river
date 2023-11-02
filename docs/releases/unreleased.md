# Unreleased

River's mini-batch methods now support pandas v2. In particular, River conforms to pandas' new sparse API.

## anomaly

- Added `anomaly.LocalOutlierFactor`, which is an online version of the LOF algorithm for anomaly detection that matches the scikit-learn implementation.
  - Made `score_one` method of `anomaly.LocalOutlierFactor` stateless
  - Defined default score for uninitialized detector
- Implementation of the `anomaly.StandardAbsoluteDeviation` algorithm, which is a uni-variate anomaly detection algorithm, based on the implementation in [PySAD](https://github.com/selimfirat/pysad/blob/master/pysad/models/knn_cad.py) (Python Streaming Anomaly Detection)

## covariance

- Added `_from_state` method to `covariance.EmpiricalCovariance` to warm start from previous knowledge.

## clustering

- Add fixes to `cluster.DBSTREAM` algorithm, including:
  - Addition of the `-` sign before the `fading_factor` in accordance with the algorithm 2 proposed by Hashler and Bolanos (2016) to allow clusters with low weights to be removed.
  - The new `micro_cluster` is added with the key derived from the maximum key of the existing micro clusters. If the set of micro clusters is still empty (`len = 0`), a new micro cluster is added with key 0.
  - `cluster_is_up_to_date` is set to `True` at the end of the `self._recluster()` function.
  - Shared density graph update timestamps are initialized with the current timestamp value
  - `neighbour_neighbours` are appended correctly to the `seed_set` when generating cluster labels
  - When building weighted adjacency matrix the algorithm accounts for possibly orphaned entries in shared density graph

## datasets

- Added `datasets.WebTraffic`, which is a dataset that counts the occurrences of events on a website. It is a multi-output regression dataset with two outputs.

## drift

- Add `drift.NoDrift` to allow disabling the drift detection capabilities of models. This detector does nothing and always returns `False` when queried whether or not a concept drift was detected.

## evaluate

- Added a `yield_predictions` parameter to `evaluate.iter_progressive_val_score`, which allows including predictions in the output.

## forest

- Simplify inner the structures of `forest.ARFClassifier` and `forest.ARFRegressor` by removing redundant class hierarchy. Simplify how concept drift logging can be accessed in individual trees and in the forest as a whole.

## proba

- Added `_from_state` method to `proba.MultivariateGaussian` to warm start from previous knowledge.

## tree

- Fix a bug in `tree.splitter.NominalSplitterClassif` that generated a mismatch between the number of existing tree branches and the number of tracked branches.
- Fix a bug in `tree.ExtremelyFastDecisionTreeClassifier` where the split re-evaluation failed when the current branch's feature was not available as a split option. The fix also enables the tree to pre-prune a leaf via the tie-breaking mechanism.

## stats

- Implementation of the incremental Kolmogorov-Smirnov statistics (at `stats.KolmogorovSmirnov`), with the option to calculate either the original KS or Kuiper's test.

## utils

- Removed `utils.dict2numpy` and `utils.numpy2dict` functions. They were not used anywhere in the library.
- `utils.TimeRolling` now works correctly if two samples with the same timestamp are added in a row.
