import datetime as dt
import time
import typing
from itertools import accumulate, cycle

from river import base, metrics, stream, utils

__all__ = ["progressive_val_score"]


def _progressive_validation(
    dataset: base.typing.Stream,
    model,
    metric: metrics.Metric,
    checkpoints: typing.Iterator[int],
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    measure_time=False,
    measure_memory=False,
):

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(
            f"{metric.__class__.__name__} metric is not compatible with {model}"
        )

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    if utils.inspect.isclassifier(model) and not metric.requires_labels:
        pred_func = model.predict_proba_one

    preds = {}

    next_checkpoint = next(checkpoints, None)
    n_total_answers = 0
    if measure_time:
        start = time.perf_counter()

    for i, x, y in stream.simulate_qa(dataset, moment, delay, copy=True):

        # Question
        if y is None:
            preds[i] = pred_func(x=x)
            continue

        # Answer
        y_pred = preds.pop(i)
        if y_pred != {} and y_pred is not None:
            metric.update(y_true=y, y_pred=y_pred)
        model.learn_one(x=x, y=y)

        # Update the answer counter
        n_total_answers += 1
        if n_total_answers == next_checkpoint:
            if isinstance(metric, metrics.Metrics):
                results = {m.__class__.__name__: m.get() for m in metric}
            else:
                results = {metric.__class__.__name__: metric.get()}
            results["Step"] = n_total_answers
            if measure_time:
                now = time.perf_counter()
                results["Time"] = dt.timedelta(seconds=now - start)
            if measure_memory:
                results["Memory"] = model._memory_usage
            yield results
            next_checkpoint = next(checkpoints, None)


def progressive_val_score(
    dataset: base.typing.Stream,
    model,
    metric: metrics.Metric,
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    print_every=0,
    show_time=False,
    show_memory=False,
    **print_kwargs,
) -> metrics.Metric:
    """Evaluates the performance of a model on a streaming dataset.

    This method is the canonical way to evaluate a model's performance. When used correctly, it
    allows you to exactly assess how a model would have performed in a production scenario.

    `dataset` is converted into a stream of questions and answers. At each step the model is either
    asked to predict an observation, or is either updated. The target is only revealed to the model
    after a certain amount of time, which is determined by the `delay` parameter. Note that under
    the hood this uses the `stream.simulate_qa` function to go through the data in arrival order.

    By default, there is no delay, which means that the samples are processed one after the other.
    When there is no delay, this function essentially performs progressive validation. When there
    is a delay, then we refer to it as delayed progressive validation.

    It is recommended to use this method when you want to determine a model's performance on a
    dataset. In particular, it is advised to use the `delay` parameter in order to get a reliable
    assessment. Indeed, in a production scenario, it is often the case that ground truths are made
    available after a certain amount of time. By using this method, you can reproduce this scenario
    and therefore truthfully assess what would have been the performance of a model on a given
    dataset.

    Parameters
    ----------
    dataset
        The stream of observations against which the model will be evaluated.
    model
        The model to evaluate.
    metric
        The metric used to evaluate the model's predictions.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected to take
        as input a `dict` of features. If `None`, then the observations are implicitly timestamped
        in the order in which they arrive.
    delay
        The amount to wait before revealing the target associated with each observation to the
        model. This value is expected to be able to sum with the `moment` value. For instance, if
        `moment` is a `datetime.date`, then `delay` is expected to be a `datetime.timedelta`. If a
        callable is passed, then it is expected to take as input a `dict` of features and the
        target. If a `str` is passed, then it will be used to access the relevant field from the
        features. If `None` is passed, then no delay will be used, which leads to doing standard
        online validation.
    print_every
        Iteration number at which to print the current metric. This only takes into account the
        predictions, and not the training steps.
    show_time
        Whether or not to display the elapsed time.
    show_memory
        Whether or not to display the memory usage of the model.
    print_kwargs
        Extra keyword arguments are passed to the `print` function. For instance, this allows
        providing a `file` argument, which indicates where to output progress.

    Examples
    --------

    Take the following model:

    >>> from river import linear_model
    >>> from river import preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression()
    ... )

    We can evaluate it on the `Phishing` dataset as so:

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics

    >>> evaluate.progressive_val_score(
    ...     model=model,
    ...     dataset=datasets.Phishing(),
    ...     metric=metrics.ROCAUC(),
    ...     print_every=200
    ... )
    [200] ROCAUC: 0.897995
    [400] ROCAUC: 0.920896
    [600] ROCAUC: 0.931339
    [800] ROCAUC: 0.939909
    [1,000] ROCAUC: 0.947417
    [1,200] ROCAUC: 0.950304
    ROCAUC: 0.950363

    We haven't specified a delay, therefore this is strictly equivalent to the following piece
    of code:

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LogisticRegression()
    ... )

    >>> metric = metrics.ROCAUC()

    >>> for x, y in datasets.Phishing():
    ...     y_pred = model.predict_proba_one(x)
    ...     metric = metric.update(y, y_pred)
    ...     model = model.learn_one(x, y)

    >>> metric
    ROCAUC: 0.950363

    When `print_every` is specified, the current state is printed at regular intervals. Under
    the hood, Python's `print` method is being used. You can pass extra keyword arguments to
    modify its behavior. For instance, you may use the `file` argument if you want to log the
    progress to a file of your choice.

    >>> with open('progress.log', 'w') as f:
    ...     metric = evaluate.progressive_val_score(
    ...         model=model,
    ...         dataset=datasets.Phishing(),
    ...         metric=metrics.ROCAUC(),
    ...         print_every=200,
    ...         file=f
    ...     )

    >>> with open('progress.log') as f:
    ...     for line in f.read().splitlines():
    ...         print(line)
    [200] ROCAUC: 0.94
    [400] ROCAUC: 0.946969
    [600] ROCAUC: 0.9517
    [800] ROCAUC: 0.954238
    [1,000] ROCAUC: 0.958207
    [1,200] ROCAUC: 0.96002

    Note that the performance is slightly better than above because we haven't used a fresh
    copy of the model. Instead, we've reused the existing model which has already done a full
    pass on the data.

    >>> import os; os.remove('progress.log')

    References
    ----------
    [^1]: [Beating the Hold-Out: Bounds for K-fold and Progressive Cross-Validation](http://hunch.net/~jl/projects/prediction_bounds/progressive_validation/coltfinal.pdf)
    [^2]: [Grzenda, M., Gomes, H.M. and Bifet, A., 2019. Delayed labelling evaluation for data streams. Data Mining and Knowledge Discovery, pp.1-30](https://link.springer.com/content/pdf/10.1007%2Fs10618-019-00654-y.pdf)

    """

    checkpoints = _progressive_validation(
        dataset,
        model,
        metric,
        checkpoints=accumulate(cycle([print_every])) if print_every else iter([]),
        moment=moment,
        delay=delay,
        measure_time=show_time,
        measure_memory=show_memory,
    )

    for checkpoint in checkpoints:

        msg = f"[{checkpoint['Step']:,d}] {metric}"
        if show_time:
            msg += f" – {checkpoint['Time']}"
        if show_memory:
            msg += f" – {checkpoint['Memory']}"
        print(msg, **print_kwargs)

    return metric
