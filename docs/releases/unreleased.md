# Unreleased

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

- Reimplemented the inner `expected_mutual_info` routine (used by `metrics.AdjustedMutualInfo`) in Rust. The Cython sources are removed and the new implementation is roughly twice as fast as the old one across all tested contingency-table sizes.
- Reimplemented `metrics.RollingROCAUC` and `metrics.RollingPRAUC` in Rust. The C++ implementation is removed. Output is bit-identical to the C++ version on all tested inputs and a latent bug in `revert()` with a non-default `pos_val` is also fixed.

## utils

- Reimplemented `utils.VectorDict` (and the helper functions `euclidean_distance_dict`, `euclidean_distance_tuple`, `lazy_search_euclidean`) in Rust. The Cython sources are removed; the public API is unchanged. Element-wise operations are faster across the board: `vec + scalar` and `vec * scalar` are ~18% faster on 20-key dicts and ~14% faster on 1000-key dicts; `vec + vec` is 4-5% faster, `vec @ vec` (dot product) is 4-10% faster. The constructor and `__setitem__` are within 1-4% of the Cython baseline (~2 ns absolute, dominated by PyO3 object-allocation overhead).

## tree

- Fixed `MondrianNodeClassifier.replant` not copying the `counts` attribute when promoting a leaf to a branch, leaving the new branch with `n_samples != 0` but empty class counts. The fix mirrors the regressor's `_mean` copy and matches the reference [`onelearn`](https://github.com/onelearn/onelearn) implementation. Addresses [#1823](https://github.com/online-ml/river/issues/1823).
- Fixed Mondrian tree leaf nodes losing their bounding box ranges during splits. Previously, when a leaf was split, the new child nodes did not inherit the `memory_range_min` and `memory_range_max` attributes, which caused incorrect range extension calculations. Fixes [#1801](https://github.com/online-ml/river/issues/1801)
- Fixed `MondrianNodeClassifier.replant` copying min and max bounds by reference instead of by value during a split. The fix ensures these arrays are explicitly copied by value so the bounds are correctly preserved. Fixed [#1834](https://github.com/online-ml/river/issues/1834)
- Skipped the expensive `range_extension_c` call for pure nodes in the Mondrian classifier's downward pass when `split_pure=False` (default). Benchmarks show ~3–5% speedup on datasets with 50+ features.
