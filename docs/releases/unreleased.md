# Unreleased

## drift

- Added `Shewhart`, the classical individuals control chart. Each sample is compared against fixed limits set at `k` standard deviations from a center line estimated over the warm-up window. Unlike `EWMA` and `CUSUM` it accumulates no evidence, so it is the fastest to react to a large abrupt shift and the noisiest on ordinary variation.

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.

## tree

- The Rust-backed `tree.HoeffdingTreeClassifier`, along with ensembles such as `forest.ARFClassifier`, no longer leaves its leaves permanently switched off when one class dominates. Leaves continue learning and can split again when appropriate, restoring the behaviour of version 0.26.1.
