# 0.21.1 - 2024-03-28

This release should fix some of the installation issues when building the River wheel from scratch.

## anomaly

- Added `PredictiveAnomalyDetection`, a semi-supervised technique that employs a predictive model for anomaly detection.

## cluster

- Added `ODAC` (Online Divisive-Agglomerative Clustering) for clustering time series.

## drift

- Added `FHDDM` drift detector.
- Added a `iter_polars` function to iterate over the rows of a polars DataFrame.

## neighbors

- Simplified `neighbors.SWINN` to avoid recursion limit and pickling issues.
