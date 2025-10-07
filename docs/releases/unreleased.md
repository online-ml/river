# Unreleased

## sketch

- Add the `sketch.NUnique` class. It was previoulsy in the `stats` module. This sketch estimates the number of unique elements in a stream.

## stats

- Added `update_many` method to `stats.PearsonCorr`.
- Moved `stats.NUnique` to the `sketch` module, as it is more of a sketch than a statistical indicator.
