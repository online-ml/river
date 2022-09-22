# Unreleased

- Introducing the `sketch` module with summarization tools and data sketches working in a streaming fashion!

## sketch

- Move `misc.Histogram` to `sketch.Histogram`.
- Move `stats.LossyCount` to `sketch.HeavyHitters` and update its API to better match `collections.Counter`.
- Added missing return `self` in `HeavyHitters`.
- Added the Count-Min Sketch (`sketch.Counter`) algorithm for approximate element counting.
