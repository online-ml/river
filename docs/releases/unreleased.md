# Unreleased

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.

## tree

- The Rust-backed `tree.HoeffdingTreeClassifier`, along with ensembles such as `forest.ARFClassifier`, no longer leaves its leaves permanently switched off when one class dominates. Leaves continue learning and can split again when appropriate, restoring the behaviour of version 0.26.1.
- `tree.HoeffdingTreeClassifier` and `forest.ARFClassifier` no longer crash on string feature values. Models whose first sample has non-numeric values use the Python implementation, which handles them as nominal features, as in version 0.26.1.
