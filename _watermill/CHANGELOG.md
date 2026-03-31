# Release

## 0.1.1 2023-02-06
- Fix iconsistent initialisation of `quantile::Quantile` estimates


## 0.1.0 - 2022-09-12

- `online-statistics` is `watermill` now


---------
# Archived

## 0.2.6 - 2022-09-09

- Change github owner from `AdilZouitine` to `online-ml`

## 0.2.5 - 2022-09-09

- Fix panic error ` 'called `Option::unwrap()` on a `None` value'` on `quantile::Quantile`

## 0.2.4 - 2022-08-27

- `sorted_window::SortedWindow` is public now.

## 0.2.3 - 2022-08-27

- `sorted_window::SortedWindow` is Serializable, Deserializable, therefore `iqr::RollingIQR`, `quantile::RollingQuantile`, `maximum::RollingMax`, `minimum::RollingMin`, `ptp::RollingPeakToPeak` is Serializable, Deserializable too

## 0.2.2 - 2022-08-23

- Fix `attempt to subtract with overflow` for `iqr::RollingIQR` and `quantile::RollingQuantile`

## 0.2.1 - 2022-08-22

- Fix ` Out of bounds access` for `iqr::RollingIQR` and `quantile::RollingQuantile`

## 0.2.0 - 2022-08-22

- Added `iqr::RollingIQR`

## 0.1.0 - 2022-07-23

- `watermill` is born 🎉 