# Unreleased

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, preserve its existing smoothing behavior, and keep online and mini-batch learning finite and consistent with absent or collinear features.
