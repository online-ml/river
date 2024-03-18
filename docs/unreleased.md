# Unreleased

## anomaly

- Added `PredictiveAnomalyDetection`, a semi-supervised technique that employs a predictive model for anomaly detection.

## drift

- Added `FHDDM` drift detector.
- Added a `iter_polars` function to iterate over the rows of a polars DataFrame.

## neighbors

- Simplified `neighbors.SWINN` to avoid recursion limit and pickling issues.
