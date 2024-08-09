# Unreleased

## drift

- Added `FHDDM` drift detector.
- Added a `iter_polars` function to iterate over the rows of a polars DataFrame.

## neighbors

- Simplified `neighbors.SWINN` to avoid recursion limit and pickling issues.

## decomposition

- Added `decomposition.OnlineSVD` class to perform Singular Value Decomposition.
- Added `decomposition.OnlinePCA` class to perform Principal Component Analysis.
- Added `decomposition.OnlineDMD` class to perform Dynamic Mode Decomposition.
- Added `decomposition.OnlineDMDwC` class to perform Dynamic Mode Decomposition with Control.

## preprocessing

- Added `preprocessing.Hankelizer` class to perform Hankelization of data stream.
