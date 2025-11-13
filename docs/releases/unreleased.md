# Unreleased

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Added handlying for division by zero in `tree.hoeffding_tree` for leaf size estimation.
- Changed the calculation of the Kuiper statistic in `base.KolmogorovSmirnov` to correspond to the reference implementation. The Kuiper statistic uses the difference between the maximum value and the minimum value.
