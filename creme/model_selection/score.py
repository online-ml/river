"""Model evaluation and selection."""
import bisect
import collections
import datetime as dt
import math
import time

from .. import base
from .. import utils


__all__ = ['progressive_val_score']


class Memento(collections.namedtuple('Memento', 'x y y_pred t_expire')):

    def __lt__(self, other):
        return self['t_expire'] < other['t_expire']


def progressive_val_score(X_y, model, metric, moment=None, delay=None, print_every=math.inf,
                          show_time=False, show_memory=False):
    """A variant of online scoring where the targets are revealed with a delay.

    ``X_y`` is converted into a question and answer where the model is asked to predict an
    observation. The target is only revealed to the model after a certain amount given by
    ``delay``.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        model (base.Estimator)
        metric (metrics.Metric)
        moment (callable or str): The attribute used for measuring time. If a callable
            is passed, then it is expected to take as input a `dict` of features. If ``None``, then
            the observations are implicitely timestamped in the order in which they arrive.
        delay (callable or str or datetime.timedelta or int): The amount to wait before revealing
            the target associated with each observation to the model. This value is expected to be
            able to sum with the ``moment`` value. For instance, if ``moment`` is a
            `datetime.date`, then ``delay`` is expected to be a `datetime.timedelta`. If a callable
            is passed, then it is expected to take as input a `dict` of features and the target. If
            a `str` is passed, then it will be used to access the relevant field from the features.
            If ``None`` is passed, then no delay will be used, which leads to doing standard online
            validation.
        print_every (int): Iteration number at which to print the current metric. This only takes
            into account the predictions, and not the training steps.
        show_time (bool): Whether or not to display the elapsed time.
        show_memory (bool): Whether or not to display the memory usage of the model.

    Returns:
        metrics.Metric

    References:
        1. `Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation <http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf>`_
        2. `Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30. <https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf>`_

    """

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f'{metric.__class__.__name__} metric is not compatible with {model}')

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    is_classifier = isinstance(utils.estimator_checks.guess_model(model), base.Classifier)
    if is_classifier and not metric.requires_labels:
        pred_func = model.predict_proba_one

    # Determine how to insert mementos into the queue
    queue = (
        lambda q, el: bisect.insort(q, el)
        if callable(delay) or isinstance(delay, str) else
        lambda q, el: q.append(el)
    )

    # Coerce moment to a function
    if moment is None:
        moment_func = lambda i, _: i
    elif isinstance(moment, str):
        moment_func = lambda _, x: x[moment]
    elif callable(moment):
        moment_func = lambda _, x: moment(x)

    # Coerce delay to a function
    if delay is None:
        delay_func = lambda _, __: 0
    elif isinstance(delay, str):
        delay_func = lambda x, _: x[delay]
    elif not callable(delay):
        delay_func = lambda _, __: delay

    mementos = []
    n_total_answers = 0

    if show_time:
        start = time.perf_counter()

    for i, (x, y) in enumerate(X_y):

        t = moment_func(i, x)
        d = delay_func(x, y)

        while mementos:

            # Get the oldest answer
            x_old, y_old, y_pred_old, t_expire = mementos[0]

            # If the oldest answer isn't old enough then stop
            if t_expire > t:
                break

            # Else update the metric and the model and remove the oldest answer
            if y_pred_old != {} and y_pred_old is not None:
                metric.update(y_true=y_old, y_pred=y_pred_old)
            model.fit_one(x=x_old, y=y_old)
            del mementos[0]

            # Update the answer counter
            n_total_answers += 1
            if not n_total_answers % print_every:
                msg = f'[{n_total_answers:,d}] {metric}'
                if show_time:
                    now = time.perf_counter()
                    msg += f' – {dt.timedelta(seconds=int(now - start))}'
                if show_memory:
                    msg += f' – {model._memory_usage}'
                print(msg)

        # Make a prediction
        y_pred = pred_func(x=x)

        # Store the answer for the future
        queue(mementos, Memento(x, y, y_pred, t + d))

    # Update the metric with the remaining answers
    for _, y_old, y_pred_old, __ in mementos:
        metric.update(y_true=y_old, y_pred=y_pred_old)

    return metric
