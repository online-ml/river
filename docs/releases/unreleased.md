# Unreleased

## linear_model

- Fixed `BayesianLinearRegression` to use `alpha` as the prior precision, retain the prior during smoothing, and keep online and mini-batch learning finite and consistent with absent or collinear features.
- Clarified that with smoothing enabled, `alpha` acts as a ridge penalty that is not washed out by the data, and that a small `alpha` (or standardised features) is recommended.
