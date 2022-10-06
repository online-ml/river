# Unreleased

- Introducing the `sketch` module with summarization tools and data sketches working in a streaming fashion!

## proba

- Added `proba.Beta`.
- Added a `sample` method to each distribution.
- Replaced the `pmf` and `pdf` methods with a `__call__` method.

## sketch

- Move `misc.Histogram` to `sketch.Histogram`.
- Move `stats.LossyCount` to `sketch.HeavyHitters` and update its API to better match `collections.Counter`.
- Added missing return `self` in `HeavyHitters`.
- Added the Count-Min Sketch (`sketch.Counter`) algorithm for approximate element counting.
- Added an implementation of Bloom filter (`sketch.Set`) to provide approximate set-like operations.
