# Unreleased

## drift

- Added `CUSUM`, the original tabular cumulative-sum control chart (Page 1954), with one-sided and two-sided modes. The reference mean and deviation are estimated over the warm-up window and then frozen, and the slack and decision interval are expressed as multiples of that deviation.

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.

## tree

- The Rust-backed `tree.HoeffdingTreeClassifier`, along with ensembles such as `forest.ARFClassifier`, no longer leaves its leaves permanently switched off when one class dominates. Leaves continue learning and can split again when appropriate, restoring the behaviour of version 0.26.1.
