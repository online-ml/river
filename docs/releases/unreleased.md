# Unreleased

- Introducing the `collections` module with summarization tools and data sketches working in a streaming fashion!

## collections

- Rename `river.misc` to `river.collections`.
- Move `stats.LossyCount` to `river.collections.HeavyHitters` and update its API to better match `collections.Counter` from the standard Python library.
- Added missing return `self` in `HeavyHitters`.
- Added the Count-Min Sketch (`river.collections.Counter`) algorithm for approximate element counting.
