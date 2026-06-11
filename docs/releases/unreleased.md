# Unreleased

## datasets

- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots
- Added the BETH dataset for labeled system process events.

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.

## tree

- Added handling for division by zero in `tree.hoeffding_tree` for leaf size estimation.

## decomposition

- Added new `decomposition` module with online decomposition methods:
  - `OnlineSVD` — Online Singular Value Decomposition based on Brand (2006).
  - `OnlineSVDZhang` — Online SVD with automatic reorthogonalization based on Zhang (2022).
  - `OnlinePCA` — Online Principal Component Analysis based on Eftekhari et al. (2019).
  - `OnlineDMD` — Online Dynamic Mode Decomposition based on Zhang et al. (2019).
  - `OnlineDMDwC` — Online DMD with Control inputs.

## preprocessing

- Added `Hankelizer` transformer for time-delayed embedding of feature spaces.

## build

- Added Python 3.14 wheel builds and updated PyO3 for 3.14 support.
