from . import base
from . import pipeline


__all__ = ['iter_online_score', 'online_score']


def _identity(y_pred):
    return y_pred


def _get_probas_as_list(y_pred):
    return list(y_pred.values())


def _get_class(y_pred):
    return max(y_pred, key=y_pred.get)


def online_score(X_y, model, metric, use_proba=False):
    """Computes the online score of a model given a stream of data and a metric.

    """

    get_pred = _identity

    if isinstance(model, pipeline.Pipeline):

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


def iter_online_score(stream, model, scoring, k=1):

    y_true, y_pred = [], []

    for i, (x, y) in enumerate(stream):
        y_true.append(y)
        y_pred.append(model.fit_one(x, y))
        if i % k == 0:
            yield scoring(y_true, y_pred)
