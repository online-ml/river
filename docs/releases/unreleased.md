# Unreleased

## datasets
- Fixed download in Insects dataset. The datasets incremental_abrupt_imbalanced, incremental_imbalanced, incremental_reoccurring_imbalanced and out-of-control are not supported anymore.
- Refactored `benchmarks` and added plotly dependency for interactive plots
- Added the BETH dataset for labeled system process events.

## build

- Added Python 3.14 wheel builds and updated PyO3 for 3.14 support.
- Replaced poetry with uv for dependency management.
