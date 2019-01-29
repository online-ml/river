"""
Utilities for measuring the performance of online learning algorithms.
"""
from . import base
from . import compose


__all__ = ['online_score']


def _identity(y_pred):
    return y_pred


def _get_probas_as_list(y_pred):
    return list(y_pred.values())


def _get_class(y_pred):
    return max(y_pred, key=y_pred.get)


def online_score(X_y, model, metric, use_proba=False):
    """Computes the online score of a model given a stream of data and a metric.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        model (estimator)
        metric (callable)
        use_proba (bool)

    Returns:
        float

    """

    get_pred = _identity

    if isinstance(model, compose.Pipeline):

        if isinstance(model._final_estimator, base.MultiClassifier):
            get_pred = _get_probas_as_list if use_proba else _get_class

    else:

        if isinstance(model, base.MultiClassifier):
            get_pred = _get_probas_as_list if use_proba else _get_class

    y_true, y_pred = [], []

    for x, y in X_y:
        y_true.append(y)
        y_pred.append(get_pred(model.fit_one(x, y)))

    return metric(y_true, y_pred)
