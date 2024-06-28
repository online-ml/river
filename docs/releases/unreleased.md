# Unreleased

This release makes Polars an optional dependency instead of a required one.

## cluster

- Added `ODAC` (Online Divisive-Agglomerative Clustering) for clustering time series.

## forest

- Fix error in `forest.ARFClassifer` and `forest.ARFRegressor` where the algorithms would crash in case the number of features available for learning went below the value of the `max_features` parameter (#1560).
