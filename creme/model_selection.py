"""
Utilities for measuring the performance of online learning algorithms.
"""
from . import base


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
    pred_func = model.predict_one
    if isinstance(model, base.Classifier) and not metric.requires_labels:
        pred_func = model.predict_proba_one

    # Train the model and use the out-of-fold predictions to update the metric
    for x, y in X_y:
        y_pred = pred_func(x)
        model = model.fit_one(x, y)
        if y_pred != {} and y_pred is not None:
            metric.update(y, y_pred)

    return metric
