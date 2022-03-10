# Unreleased

- Moved all metrics in `metrics.cluster` except `metrics.Silhouette` to [river-extra](https://github.com/online-ml/river-extra).

## rules

- Now AMRules' rules representation show a default consequent: the target mean.
- AMRules's `debug_one` explicitly indicates the prediction strategy used by each rule.