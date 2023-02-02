from __future__ import annotations

import bisect
import collections
import datetime as dt
import typing
from copy import deepcopy

from river import base

__all__ = ["simulate_qa"]


class Memento(collections.namedtuple("Memento", "i x y kwargs t_expire")):
    def __lt__(self, other):
        return self.t_expire < other.t_expire


def simulate_qa(
    dataset: base.typing.Dataset,
    moment: str | typing.Callable[[dict], dt.datetime] | None,
    delay: str | int | dt.timedelta | typing.Callable | None,
    copy: bool = True,
):
    """Simulate a time-ordered question and answer session.

    This method allows looping through a dataset in the order in which it arrived. Indeed, it
    usually is the case that labels arrive after features. Being able to go through a dataset in
    arrival order enables assessing a model's performance in a reliable manner. For instance, the
    `evaluate.progressive_val_score` is a high-level method that can be used to score a model on a
    dataset. Under the hood it uses this method to determine the correct arrival order.

    Parameters
    ----------
    dataset
        A stream of (features, target) tuples.
    moment
        The attribute used for measuring time. If a callable is passed, then it is expected
        to take as input a `dict` of features. If `None`, then the observations are implicitly
        timestamped in the order in which they arrive. If a `str` is passed, then it will be
        used to obtain the time from the input features.
    delay
        The amount of time to wait before revealing the target associated with each
        observation to the model. This value is expected to be able to sum with the `moment`
        value. For instance, if `moment` is a `datetime.date`, then `delay` is expected to be a
        `datetime.timedelta`. If a callable is passed, then it is expected to take as input a
        `dict` of features and the target. If a `str` is passed, then it will be used to access
        the relevant field from the features. If `None` is passed, then no delay will be used,
        which leads to doing standard online validation. If a scalar is passed, such an `int`
        or a `datetime.timedelta`, then the delay is constant.
    copy
        If `True`, then a separate copy of the features are yielded the second time
        around. This ensures that inadvertent modifications in downstream code don't have any
        effect.

    Examples
    --------

    The arrival delay isn't usually indicated in a dataset, but it might be able to be inferred
    from the features. As an example, we'll simulate the departure and arrival time of taxi
    trips. Let's first create a time table which records the departure time and the duration of
    seconds of several taxi trips.

    >>> import datetime as dt
    >>> time_table = [
    ...     (dt.datetime(2020, 1, 1, 20,  0, 0),  900),
    ...     (dt.datetime(2020, 1, 1, 20, 10, 0), 1800),
    ...     (dt.datetime(2020, 1, 1, 20, 20, 0),  300),
    ...     (dt.datetime(2020, 1, 1, 20, 45, 0),  400),
    ...     (dt.datetime(2020, 1, 1, 20, 50, 0),  240),
    ...     (dt.datetime(2020, 1, 1, 20, 55, 0),  450)
    ... ]

    We can now create a streaming dataset where the features are the departure dates and the
    targets are the durations.

    >>> dataset = (
    ...     ({'date': date}, duration)
    ...     for date, duration in time_table
    ... )

    Now, we can use `simulate_qa` to iterate over the events in the order in which they are
    meant to occur.

    >>> delay = lambda _, y: dt.timedelta(seconds=y)

    >>> for i, x, y in simulate_qa(dataset, moment='date', delay=delay):
    ...     if y is None:
    ...         print(f'{x["date"]} - trip #{i} departs')
    ...     else:
    ...         arrival_date = x['date'] + dt.timedelta(seconds=y)
    ...         print(f'{arrival_date} - trip #{i} arrives after {y} seconds')
    2020-01-01 20:00:00 - trip #0 departs
    2020-01-01 20:10:00 - trip #1 departs
    2020-01-01 20:15:00 - trip #0 arrives after 900 seconds
    2020-01-01 20:20:00 - trip #2 departs
    2020-01-01 20:25:00 - trip #2 arrives after 300 seconds
    2020-01-01 20:40:00 - trip #1 arrives after 1800 seconds
    2020-01-01 20:45:00 - trip #3 departs
    2020-01-01 20:50:00 - trip #4 departs
    2020-01-01 20:51:40 - trip #3 arrives after 400 seconds
    2020-01-01 20:54:00 - trip #4 arrives after 240 seconds
    2020-01-01 20:55:00 - trip #5 departs
    2020-01-01 21:02:30 - trip #5 arrives after 450 seconds

    This function is extremely practical because it provides a reliable way to evaluate the
    performance of a model in a real scenario. Indeed, it allows to make predictions and
    perform model updates in exactly the same manner that would happen live. For instance, it
    is used in `evaluate.progressive_val_score`, which is a higher level function for
    evaluating models in an online manner.

    """

    # Determine how to insert mementos into the queue
    queue = (
        (lambda q, el: bisect.insort(q, el))
        if callable(delay) or isinstance(delay, str)
        else (lambda q, el: q.append(el))
    )

    # Coerce moment to a function
    get_moment = (
        (lambda _, x: x[moment])
        if isinstance(moment, str)
        else (lambda _, x: moment(x))  # type: ignore
        if callable(moment)
        else (lambda i, _: i)  # type: ignore
    )

    # Coerce delay to a function
    get_delay = (
        (lambda i, _: 0)
        if delay is None
        else (lambda x, _: x[delay])
        if isinstance(delay, str)
        else (lambda _, __: delay)  # type: ignore
        if not callable(delay)
        else delay  # type: ignore
    )

    mementos: list[Memento] = []

    kwargs: list

    for i, (x, y, *kwargs) in enumerate(dataset):
        kwargs = kwargs[0] if kwargs else None

        t = get_moment(i, x)
        d = get_delay(x, y)  # type: ignore

        while mementos:
            # Get the oldest answer
            i_old, x_old, y_old, kwargs_old, t_expire = mementos[0]

            # If the oldest answer isn't old enough then stop
            if t_expire > t:
                break

            # Reveal the ground truth and pop the observation from the queue
            yield (i_old, x_old, y_old, kwargs_old) if kwargs_old else (
                i_old,
                x_old,
                y_old,
            )
            del mementos[0]

        queue(mementos, Memento(i, x, y, kwargs, t + d))
        if copy:
            x = deepcopy(x)
        yield (i, x, None, kwargs) if kwargs else (i, x, None)

    for memento in mementos:
        yield (memento.i, memento.x, memento.y, memento.kwargs) if memento.kwargs else (
            memento.i,
            memento.x,
            memento.y,
        )
