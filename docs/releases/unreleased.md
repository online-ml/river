# Unreleased

## datasets
- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.

## tree

- Added handling for division by zero in `tree.hoeffding_tree` for leaf size estimation.

## neighbors

- Added function in nearest-neighbor engines to gather relevant classes/targets from the window.
- Added a virtual function to the base engine class; New NN engines need to override `refresh_targets` function
  - Classifier KNN now calls this engine-specific function under `clean_up_classes()`
