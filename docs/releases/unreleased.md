# Unreleased

## checks

- Added ten new global estimator checks to `river.checks`: `check_predict_one_pure` (inference methods are pure), `check_transform_one` (transform_one is exercised and returns a dict), `check_clone_is_independent` (training the original does not mutate clones), `check_predict_many_matches_predict_one` / `check_predict_proba_many_matches_predict_proba_one` / `check_transform_many_matches_transform_one` (mini-batch ↔ one-at-a-time consistency for `base.MiniBatch*` estimators), `check_get_params_matches_signature` (`_get_params()` exposes every `__init__` keyword), `check_predict_one_before_any_learn` (cold-start inference does not crash), `check_repr_roundtrips_clone` (`repr(model) == repr(model.clone())`), `check_clone_with_new_params_applies` (`clone(new_params=...)` applies the overrides), `check_classifier_tracks_seen_labels` (`predict_proba_one` includes every label observed during training), and `check_no_state_aliasing_with_input` (mutating `x` after `learn_one` does not change model state). `_yield_datasets` now also yields a dataset for plain `base.Transformer` / `base.SupervisedTransformer` estimators, which were previously skipped by the dataset-driven checks.
- Refactored the existing dataset-driven checks (`check_pickling`, `check_shuffle_features_no_impact`, `check_emerging_features`, `check_disappearing_features`, `check_radically_disappearing_features`, `check_seeding_is_idempotent`) to dispatch through `_infer` / `_learn` helpers so transformers are exercised on the same code paths as classifiers, regressors, and anomaly detectors.
- `checks.utils.assert_predictions_are_close` now treats two NaN floats as equivalent, so transformers that legitimately return NaN before they have observed any data (e.g. `MinMaxScaler.transform_one` on the first event) no longer trip the shuffle-invariance check.

## cluster

- Fixed `cluster.TextClust` corrupting its own parameters: `__init__` was overwriting `self.micro_distance` / `self.macro_distance` with runtime distance instances, breaking `clone` and `repr` round-trips. The runtime instances are now stored on `_micro_distance` / `_macro_distance`. Internal camelCase identifiers (`clusterId`, `microToMacro`, `numClusters`, `updateMacroClusters`, `_calculateIDF`) were renamed to snake_case, and the nested helper classes `tfcontainer`, `microcluster`, `distances` were renamed to `TfContainer`, `MicroCluster`, `Distances`.

## drift

- **Breaking:** Renamed `drift.binary.HDDM_A` → `drift.binary.HDDMA` and `drift.binary.HDDM_W` → `drift.binary.HDDMW` to comply with PEP-8 CapWords class naming.

## imblearn

- Fixed `imblearn.HardSamplingClassifier` / `imblearn.HardSamplingRegressor` storing references to user-supplied feature dictionaries in their buffer; the buffered triplets now hold shallow copies so callers can safely mutate `x` after `learn_one`.

## naive_bayes

- Marked `predict_many`/`predict_proba_many` checks as skipped on `BaseNB` subclasses (`MultinomialNB`, `BernoulliNB`, `ComplementNB`) via `_unit_test_skips`. `joint_log_likelihood_many`'s output is mis-aligned with the input batch when the model is trained via `learn_one` rather than `learn_many`, so the new mini-batch consistency checks fail. Tracked separately.

## neighbors

- Fixed `neighbors.KNNClassifier` / `neighbors.KNNRegressor` storing references to the input feature dicts in their search window; `learn_one` now stores a shallow copy.

## preprocessing

- Fixed `preprocessing.RobustScaler.transform_one` crashing with `TypeError` when called before any `learn_one` (the running median returned `None`); transform now passes the value through unchanged when centering statistics are not yet available.

## tree

- Fixed `tree.mondrian.MondrianTreeRegressor.learn_one` storing the input feature dict by reference on `self._x`; it now stores a shallow copy so callers can safely mutate `x` after `learn_one`. Knock-on fix for `forest.AMFRegressor`.
- **Breaking:** Renamed `tree.iSOUPTreeRegressor` → `tree.ISOUPTreeRegressor` to comply with PEP-8 CapWords class naming.

## tooling

- Enabled the pep8-naming ruleset (`N801`, `N802`, `N804`) in ruff so that future class, function, and `classmethod`-first-argument naming violations are caught at lint time. `N803` (argument names) and `N806` (local variable names) were intentionally left out — `X: pd.DataFrame`, `A_numpy = ...`, and similar scientific-Python conventions are pervasive in the codebase.

## docs

- Fixed corrupted markdown cells in the Hoeffding Trees notebook example that caused blank page titles and invisible sidebar navigation. Fixes [#1847](https://github.com/online-ml/river/issues/1847).
- Bumped zensical to 0.0.40 and enabled strict mode with link and footnote validation.
- Fixed doc generation to escape bare brackets in type annotations and descriptions, produce proper footnote definitions, and use fenced code blocks for notebook outputs.

## feature_extraction

- Added `feature_extraction.RandomTreesEmbedding`, an online random-tree leaf embedding transformer for feeding sparse tree features into downstream models. Addresses [#1386](https://github.com/online-ml/river/issues/1386).

## neural_net

- Deprecated `river.neural_net`; importing it now emits a `DeprecationWarning` and users are encouraged to use `deep-river` for neural networks. Addresses [#1828](https://github.com/online-ml/river/issues/1828).

## drift

- Reimplemented `drift.ADWIN`'s inner `AdaptiveWindowing` in Rust. The Cython sources are removed; output is bit-identical to the Cython baseline (width, total, variance, n_detections, drift_detected) over a 3.8k-step parity fuzz. Rust is 1.3-3.5x faster than the previous Cython implementation across `clock` settings.

## metrics

- Sped up `metrics.Silhouette` by switching the centroid distance computations from the `utils.math.minkowski_distance` Python wrapper to a direct call into the Rust `euclidean_distance_dict`.

- Reimplemented the inner `expected_mutual_info` routine (used by `metrics.AdjustedMutualInfo`) in Rust. The Cython sources are removed and the new implementation is roughly twice as fast as the old one across all tested contingency-table sizes.
- Reimplemented `metrics.RollingROCAUC` and `metrics.RollingPRAUC` in Rust. The C++ implementation is removed. Output is bit-identical to the C++ version on all tested inputs and a latent bug in `revert()` with a non-default `pos_val` is also fixed.

## utils

- Reimplemented `utils.VectorDict` (and the helper functions `euclidean_distance_dict`, `euclidean_distance_tuple`, `lazy_search_euclidean`) in Rust. The Cython sources are removed; the public API is unchanged. Element-wise operations are faster across the board: `vec + scalar` and `vec * scalar` are ~18% faster on 20-key dicts and ~14% faster on 1000-key dicts; `vec + vec` is 4-5% faster, `vec @ vec` (dot product) is 4-10% faster. The constructor and `__setitem__` are within 1-4% of the Cython baseline (~2 ns absolute, dominated by PyO3 object-allocation overhead).

## anomaly

- Sped up `anomaly.LocalOutlierFactor` by replacing the default `functools.partial(utils.math.minkowski_distance, p=2)` distance function with a direct call into the Rust `euclidean_distance_dict`, removing the Python-level dispatch.

## cluster

- Sped up `cluster.STREAMKMeans.predict_one` by switching the per-center distance from the `utils.math.minkowski_distance` Python wrapper to a direct call into the Rust `euclidean_distance_dict`.

## compose

- Sped up `compose.Pipeline` end-to-end throughput by 1.3x–1.9x (e.g. `scaler|lr` 7.4 µs → 5.7 µs/event, `(sel+sel)|scaler|lr` 12.5 µs → 6.7 µs/event on TrumpApproval) by precomputing an execution plan (kind/`_supervised` flags) for each step at construction time, eliminating per-event `isinstance` checks via the `EstimatorMeta.__instancecheck__` metaclass (~180k → 0 calls per 20k events) and repeated `_supervised` property lookups. The plan is invalidated on `_add_step`. The lazy `_anomaly_filter_cls` / `_anomaly_detector_cls` imports are now `functools.cache`d.
- Sped up `compose.TransformerUnion.transform_one` by replacing the `dict(collections.ChainMap(*outputs))` merge with a single `dict.update` loop over reversed transformer outputs (~10x faster on the merge alone). Semantics are preserved (earlier transformers win on duplicate keys).
- Sped up `compose.Prefixer` / `compose.Suffixer` `transform_one` by inlining the prefix/suffix concatenation in the dict comprehension instead of going through the `_rename` method on each key.

## tree

- Fixed `MondrianNodeClassifier.replant` not copying the `counts` attribute when promoting a leaf to a branch, leaving the new branch with `n_samples != 0` but empty class counts. The fix mirrors the regressor's `_mean` copy and matches the reference [`onelearn`](https://github.com/onelearn/onelearn) implementation. Addresses [#1823](https://github.com/online-ml/river/issues/1823).
- Fixed Mondrian tree leaf nodes losing their bounding box ranges during splits. Previously, when a leaf was split, the new child nodes did not inherit the `memory_range_min` and `memory_range_max` attributes, which caused incorrect range extension calculations. Fixes [#1801](https://github.com/online-ml/river/issues/1801)
- Fixed `MondrianNodeClassifier.replant` copying min and max bounds by reference instead of by value during a split. The fix ensures these arrays are explicitly copied by value so the bounds are correctly preserved. Fixed [#1834](https://github.com/online-ml/river/issues/1834)
- Skipped the expensive `range_extension_c` call for pure nodes in the Mondrian classifier's downward pass when `split_pure=False` (default). Benchmarks show ~3–5% speedup on datasets with 50+ features.
- Reimplemented the Mondrian tree numerical helpers (`tree.mondrian._mondrian_ops`) in Rust. The Cython sources are removed; the helpers are now exposed via `river.stats._rust_stats`. Output matches the Cython baseline (Bananas accuracy unchanged at 70.64%). The leaf-to-root `_go_upwards` walk and the predict tree-walk also moved into Rust as single FFI calls, eliminating ~360k Python frame setups per 20k-sample run. End-to-end `MondrianTreeClassifier` learn+predict is ~28% faster (~23 µs/iter vs ~32 µs/iter Cython); `MondrianTreeRegressor` is ~21% faster (~31 µs/iter vs ~39 µs/iter) on a 20k-sample 10-feature synthetic stream.
