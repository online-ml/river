# Unreleased

- Moved all metrics in `metrics.cluster` except `metrics.Silhouette` to [river-extra](https://github.com/online-ml/river-extra).

## dist

- A `revert` method has been added to `stats.Gaussian`.
- A `revert` method has been added to `stats.Multinomial`.
- Added `dist.TimeRolling` to measure probability distributions over windows of time.

## rules

- Now AMRules' rules representation show a default consequent: the target mean.
- AMRules's `debug_one` explicitly indicates the prediction strategy used by each rule.

## stats

- A `revert` method has been added to `stats.Var`.
