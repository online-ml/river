# Unreleased

## datasets
- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots
- Added the BETH dataset for labeled system process events.

## dummy

The `dummy` module is now fully type-annotated.

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.

## tree

- Added handling for division by zero in `tree.hoeffding_tree` for leaf size estimation.

## build

- Added Python 3.14 wheel builds and updated PyO3 for 3.14 support.
