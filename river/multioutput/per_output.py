from __future__ import annotations

import collections
import copy

from river import base, linear_model

__all__ = ["PerOutputRegressor"]


class PerOutputRegressor(base.Wrapper, base.MultiTargetRegressor, collections.UserDict):  # type:ignore[misc]
    """A multi-output model that trains one independent regressor per output.

    This model does not use the prediction of one output as a
    feature for the next. Each output is modelled by its own copy of the base regressor,
    trained independently. (This is the streaming equivalent of scikit-learn's
    `MultiOutputRegressor`).

    The set of outputs isn't known from the start in a streaming setting, new regressors are
    instantiated on the fly, one per output key encountered in `y`.

    Parameters
    ----------
    model
        The regression model used to make predictions for each target.

    Examples
    --------

    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import stream

    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_linnerud(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = multioutput.PerOutputRegressor(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(intercept_lr=0.3)
    ...     )
    ... )

    >>> metric = metrics.multioutput.MicroAverage(metrics.MAE())

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MicroAverage(MAE): 12.68377

    """

    def __init__(self, model: base.Regressor):
        super().__init__()
        self.model = model

    @property
    def _wrapped_model(self):
        return self.model

    def __getitem__(self, key):
        try:
            return collections.UserDict.__getitem__(self, key)
        except KeyError:
            collections.UserDict.__setitem__(self, key, copy.deepcopy(self.model))
            return self[key]

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y, **kwargs):
        for o, y_o in y.items():
            self[o].learn_one(x, y_o, **kwargs)

    def predict_one(self, x, **kwargs):
        return {o: reg.predict_one(x, **kwargs) for o, reg in self.items()}
