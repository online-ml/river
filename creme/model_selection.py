"""
Utilities for measuring the performance of online learning algorithms.
"""
from . import base
from . import compose
from . import metrics


__all__ = ['online_score']


def online_score(X_y, model, metric):
    """Computes the online score of a model given a stream of data and a metric.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        model (estimator)
        metric (callable)

    Returns:
        metric

    """

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f"{metric.__class__.__name__} metric can't be used to evaluate a " +
                         f'{model.__class__.__name__}')

    # Determine if predict_one or predict_proba_one should be used
    fit_predict = model.fit_predict_one
    is_classifier = isinstance(model, (base.BinaryClassifier, base.MultiClassifier))
    if is_classifier and not metric.requires_labels:
        fit_predict = model.fit_predict_proba_one

    # Train the model and use the out-of-fold predictions to update the metric
    for x, y in X_y:
        y_pred = fit_predict(x, y)
        metric.update(y, y_pred)

    return metric
