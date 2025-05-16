# Unreleased

## base

- The `tags` and `more_tags` properties of `base.Estimator` are now both a set of strings.
- The `base` module is now fully type-annotated. Some type hints have changed, but this does not impact the behaviour of the code. For instance, the regression target is now indicated as a float instead of a Number.
- `base.Ensemble`, `base.Wrapper`, and `base.WrapperEnsemble` became generic with regard to the type they encapsulate.

## neighbors

- Added `neighbors.SAMkNNClassifier` implementing the SAM-kNN Classifier
