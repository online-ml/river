# Unreleased

## ensemble

- Bug fixes in `SRPClassifier` and `SRPRegressor`.

## optim

- `optim.Adam` and `optim.RMSProp` now work with `utils.VectorDict`s as well as `numpy.ndarray`s.

## stats

- Fixed an issue where some statistics could not be printed if they had not seen any data yet.
- Implemented median absolute deviation in `stats.MAD`.
