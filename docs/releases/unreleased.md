# Unreleased

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.

## tree

- The Rust-backed `tree.HoeffdingTreeClassifier`, along with ensembles such as `forest.ARFClassifier`, no longer leaves its leaves permanently switched off when one class dominates. Leaves continue learning and can split again when appropriate, restoring the behaviour of version 0.26.1.

## calibration

- Added `calibration.CalibratedClassifier`, a wrapper that recalibrates a binary classifier's probability estimates with online Platt scaling. It fits a logistic function $\sigma(a \cdot s + b)$ on the wrapped model's score, where the score is read out-of-sample (before the wrapped model learns on the current sample), and updates $a$ and $b$ with a gradient step on the log-loss at every `learn_one`. It starts from the identity mapping ($a=1$, $b=0$), so it is a no-op until data flows in.
