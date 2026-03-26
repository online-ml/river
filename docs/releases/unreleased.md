# Unreleased

## build

- Added Python 3.14 wheel builds and updated PyO3 for 3.14 support.
- Replaced poetry with uv for dependency management.

## datasets

- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots
- Added the BETH dataset for labeled system process events.

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.
- Fixed `RollingQuantile` not storing `q` as an instance attribute, which caused `clone()` to fail.

## metrics

- Fixed `KeyError` in `Silhouette` metric when used with clusterers that haven't initialized their centers yet (e.g., `CluStream` during its warmup phase).

## tree

- Added handling for division by zero in `tree.hoeffding_tree` for leaf size estimation.
