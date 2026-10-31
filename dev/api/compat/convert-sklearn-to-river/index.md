# convert\_sklearn\_to\_river

Wraps a scikit-learn estimator to make it compatible with river.

## Parameters

- **estimator**

  *Type* → `sklearn_base.BaseEstimator`
- **classes**

  *Type* → `list[base.typing.ClfTarget] | None`

  *Default* → `None`

  Class names. Required for classifiers, and not accepted for regressors.
