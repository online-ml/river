"""
Utilities for measuring the performance of online learning algorithms.
"""
import math

from . import base
from . import stream


__all__ = ['online_score', 'online_qa_score']


def online_score(X_y, model, metric, print_every=math.inf):
    """Computes the online score of a model given a stream of data and a metric.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        model (estimator)
        metric (callable)
        print_every (int): Iteration number at which to print the current metric.

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

    # Handle the first observation separately
    x, y = next(X_y)
    y_pred = pred_func(x)
    model = model.fit_one(x, y)
    if y_pred != {} and y_pred is not None:
        metric.update(y, y_pred)

    # Train the model and use the out-of-fold predictions to update the metric
    for i, (x, y) in enumerate(X_y):
        y_pred = pred_func(x)
        model = model.fit_one(x, y)
        metric.update(y, y_pred)

        if not (i + 1) % print_every:
            print(f'[{i+1}] {metric}')

    return metric


def online_qa_score(X_y, model, metric, on, lag, print_every=math.inf):
    """A variant of online scoring where the targets are revealed with a lag.

    ``X_y`` is converted into a question and answer where the model is asked to predict an
    observation. The target is only revealed to the model after a certain amount of ``lag``. See
    `stream.simulate_qa` for more information.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        model (estimator)
        metric (callable)
        on (str): The attribute used for measuring time.
        lag (datetime.timedelta or int or float): Amount to wait before revealing the target
            associated with each observation. This value is expected to be able to sum with the
            `on` attribute.
        print_every (int): Iteration number at which to print the current metric.

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
    for i, (is_question, x, y) in enumerate(stream.simulate_qa(X_y, on=on, lag=lag)):

        # If y is None then this is a question
        if is_question:
            y_pred = pred_func(x)
            if y_pred != {} and y_pred is not None:
                metric.update(y, y_pred)

        # If not this is an answer
        else:
            model = model.fit_one(x, y)

        if not (i + 1) % print_every:
            print(f'[{i+1}] {metric}')

    return metric
