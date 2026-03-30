# Unreleased

## build

- Added Python 3.14 wheel builds and updated PyO3 for 3.14 support.
- Replaced poetry with uv for dependency management.

## datasets

- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots
- Added the BETH dataset for labeled system process events.
- Fixed `SMTP` dataset docstring: corrected the number of positive labels from 2,211 to 30 and updated the reference link.

## cluster

- Fixed DBSTREAM including noisy micro-clusters (weight below `minimum_weight`) in output clusters. They are now excluded during reclustering, matching the original paper.

## forest

- Added `max_nodes` parameter to `AMFClassifier`, `AMFRegressor`, and the underlying Mondrian tree classes. This caps the number of nodes per tree, limiting memory usage for long-running streams. Addresses [#1454](https://github.com/online-ml/river/issues/1454).

## drift

- Optimized `ADWIN` Cython internals (~18x speedup): replaced numpy arrays with C `malloc`/`memmove` arrays in `Bucket`, replaced Python `deque` with typed `list`, used bit shifts instead of `pow`, inlined `variance_in_window`, and added Cython compiler directives.

## dummy

The `dummy` module is now fully type-annotated.

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.
- Fixed `RollingQuantile` not storing `q` as an instance attribute, which caused `clone()` to fail.
- Optimized `Var.update`/`revert` and `Cov.update`/`revert` by replacing `Mean.get()` method calls with direct `_mean` attribute access and inlining property lookups (~19% speedup each).
- Optimized `KolmogorovSmirnov` treap internals: replaced class-based `Treap` with `__slots__` nodes and module-level functions, inlined lazy propagation, and eliminated builtin `max`/`min` overhead. This yields a 2.65x speedup on update/revert operations.

## compat

- Adapted sklearn compatibility layer to sklearn 1.8: replaced `_more_tags` with `__sklearn_tags__`, switched from `check_X_y`/`check_array` to `validate_data`, fixed mixin inheritance order, and updated binary classifier validation.

## metrics

- Fixed `AdjustedMutualInfo` to return 0.0 when only one class or one cluster exists, and to handle the 0/0 edge case for perfect matches with small samples, aligning with sklearn 1.8 behavior.
- Fixed `KeyError` in `Silhouette` metric when used with clusterers that haven't initialized their centers yet (e.g., `CluStream` during its warmup phase).
- Optimized `ConfusionMatrix` by inlining `_update` into `update`/`revert` (~10% speedup) and caching `total_true_positives` as an incrementally maintained counter (99% speedup on access).
- Cached `requires_labels` in `BinaryMetric.__init__` to avoid property lookup on every `update`/`revert` call.

## evaluate

- Optimized `progressive_val_score` and `iter_progressive_val_score` with a fast path for the common no-delay case. The evaluation loop now iterates the dataset directly, skipping the `simulate_qa` generator and internal prediction buffer. Combined with caching `model._supervised` and `metric.update`, this yields a **1.5x speedup** on typical workloads.

## stream

- `stream.iter_arff` now supports blank values (treated as missing values).

## preprocessing

- Add support for expected categories in `preprocessing.OneHotEncoder`, `preprocessing.OrdinalEncoder`, akin to scikit-learn API for respective encoders.
- Added a fast path in `simulate_qa` for the no-delay, no-moment case, skipping the memento queue machinery.

## base

- Added `EstimatorMeta` metaclass so that `isinstance` works transparently with pipelines. For example, `isinstance(scaler | log_reg, base.Classifier)` now returns `True`. This removes the need for `utils.inspect` helper functions (`isclassifier`, `isregressor`, etc.), which have been removed.

## proba

- Optimized `Gaussian.__call__` by inlining property accesses, and added `Gaussian.log_pdf` method that computes log-density directly without `exp`/`sqrt`. This speeds up Naive Bayes prediction in all Hoeffding Tree classifiers and their ensembles by 12–22%.

## tree

- Added `cond_log_proba` to `GaussianSplitter` and optimized `do_naive_bayes_prediction` to use direct log-probabilities, avoiding the `exp`/`log` round-trip.
- Added handling for division by zero in `tree.hoeffding_tree` for leaf size estimation.

## neighbors

- Added function in nearest-neighbor engines to gather relevant classes/targets from the window.
- Added a virtual function to the base engine class; New NN engines need to override `refresh_targets` function
  - Classifier KNN now calls this engine-specific function under `clean_up_classes()`

## utils

- The `utils` module is now fully type-checked.
- `utils.VectorDict` and `utils.SortedWindow` are now parametrised generic containers.
- `utils.VectorDict` now implements the reflected operations of addition, subtraction and multiplication.
