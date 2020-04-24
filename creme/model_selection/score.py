"""Model evaluation and selection."""
import bisect
import collections
import datetime as dt
import math
import time
import typing

from creme import base
from creme import metrics
from creme import utils
from creme import stream


__all__ = ['progressive_val_score']


def progressive_val_score(X_y: base.typing.Stream, model: base.Predictor, metric: metrics.Metric,
                          moment: typing.Union[str, typing.Callable] = None,
                          delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
                          print_every=0, show_time=False, show_memory=False) -> metrics.Metric:
    """A variant of online scoring where the targets are revealed with a delay.

    `X_y` is converted into a stream of questions and answers. At each step the model is either
    asked to predict an observation, or is either updated. The target is only revealed to the model
    after a certain amount of time, which is determined by `delay` the parameter.

    Parameters:
        X_y: The stream of observations against which the model will be evaluated.
        model: The model to evaluate.
        metric: The metric used to evaluate the model's predictions.
        moment (callable or str): The attribute used for measuring time. If a callable
            is passed, then it is expected to take as input a `dict` of features. If `None`, then
            the observations are implicitely timestamped in the order in which they arrive.
        delay: The amount to wait before revealing the target associated with each observation to
            the model. This value is expected to be able to sum with the `moment` value. For
            instance, if `moment` is a `datetime.date`, then `delay` is expected to be a
            `datetime.timedelta`. If a callable is passed, then it is expected to take as input a
            `dict` of features and the target. If a `str` is passed, then it will be used to access
            the relevant field from the features. If `None` is passed, then no delay will be
            used, which leads to doing standard online validation.
        print_every (int): Iteration number at which to print the current metric. This only takes
            into account the predictions, and not the training steps.
        show_time (bool): Whether or not to display the elapsed time.
        show_memory (bool): Whether or not to display the memory usage of the model.

    References:
        1. [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
        2. [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f'{metric.__class__.__name__} metric is not compatible with {model}')

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    is_classifier = isinstance(utils.estimator_checks.guess_model(model), base.Classifier)
    if is_classifier and not metric.requires_labels:
        pred_func = model.predict_proba_one

    preds = {}

    n_total_answers = 0
    if show_time:
        start = time.perf_counter()

    for i, x, y in stream.simulate_qa(X_y, moment, delay, copy=True):

        # Question
        if y is None:
            preds[i] = pred_func(x=x)

        # Answer
        else:
            y_pred = preds.pop(i)
            if y_pred != {} and y_pred is not None:
                metric.update(y_true=y, y_pred=y_pred)
            model.fit_one(x=x, y=y)

            # Update the answer counter
            n_total_answers += 1
            if print_every and not n_total_answers % print_every:
                msg = f'[{n_total_answers:,d}] {metric}'
                if show_time:
                    now = time.perf_counter()
                    msg += f' – {dt.timedelta(seconds=int(now - start))}'
                if show_memory:
                    msg += f' – {model._memory_usage}'
                print(msg)

    return metric
