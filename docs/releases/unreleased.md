# Unreleased

## base

- The `tags` and `more_tags` properties of `base.Estimator` are now both a set of strings.
- The `base` module is now fully type-annotated. Some type hints have changed, but this does not impact the behaviour of the code. For instance, the regression target is now indicated as a float instead of a Number.
- `base.Ensemble`, `base.Wrapper`, and `base.WrapperEnsemble` became generic with regard to the type they encapsulate.

## neighbors

- Remove the `itertools.cycle` usage from the `neighbors.ann.SWINN` search engine, as pickling `cycle` objects will not be supported anymore, starting from Python 3.14. This change has no effect from the user standpoint, as the 'cycle' usage was more of a gimmick than a necessity.
