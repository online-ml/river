# Unreleased

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.

## tree

- The Rust-backed `tree.HoeffdingTreeClassifier`, along with ensembles such as `forest.ARFClassifier`, no longer leaves its leaves permanently switched off when one class dominates. Leaves continue learning and can split again when appropriate, restoring the behaviour of version 0.26.1.

## metrics

### Fixed binary classification metrics with custom positive labels

Fixed incorrect results in Precision, Recall, F-beta, Jaccard, MCC, and ROCAUC when `pos_val` was set to a label other than `True` or `1`.

Binary metrics now preserve the original class labels when updating confusion matrices. MCC uses the correct one-vs-rest true-negative calculation, and ROCAUC correctly identifies the positive class when evaluating predictions.

Added regression tests covering custom string labels and weighted one-vs-rest predictions.
