# Unreleased

River's mini-batch methods now support pandas v2. In particular, River conforms to pandas' new sparse API.

## anomaly

- Added `anomaly.LocalOutlierFactor`, which is an online version of the LOF algorithm for anomaly detection that matches the scikit-learn implementation.
