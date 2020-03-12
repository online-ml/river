import bisect
import collections


__all__ = ['simulate_qa']


class Memento(collections.namedtuple('Memento', 'i x y t_expire')):

    def __lt__(self, other):
        return self.t_expire < other.t_expire


def simulate_qa(X_y, moment, delay):
    """Simulate a time-ordered question and answer session.

    Parameters:
        X_y (generator): A stream of (features, target) tuples.
        moment (callable or str): The attribute used for measuring time. If a callable
            is passed, then it is expected to take as input a `dict` of features. If ``None``, then
            the observations are implicitely timestamped in the order in which they arrive. If a
            ``str`` is passed, then it will be used to obtain the time from the input features.
        delay (callable or str or datetime.timedelta or int): The amount to wait before revealing
            the target associated with each observation to the model. This value is expected to be
            able to sum with the ``moment`` value. For instance, if ``moment`` is a
            `datetime.date`, then ``delay`` is expected to be a `datetime.timedelta`. If a callable
            is passed, then it is expected to take as input a `dict` of features and the target. If
            a `str` is passed, then it will be used to access the relevant field from the features.
            If ``None`` is passed, then no delay will be used, which leads to doing standard online
            validation. If a scalar is passed, such an `int` or a `datetime.timedelta`, then the
            delay is constant.

    """

    # Determine how to insert mementos into the queue
    queue = lambda q, el: q.append(el)
    if callable(delay) or isinstance(delay, str):
        queue = lambda q, el: bisect.insort(q, el)

    # Coerce moment to a function
    if moment is None:
        get_moment = lambda i, _: i
    elif isinstance(moment, str):
        get_moment = lambda _, x: x[moment]
    elif callable(moment):
        get_moment = lambda _, x: moment(x)

    # Coerce delay to a function
    if delay is None:
        get_delay = lambda _, __: 0
    elif isinstance(delay, str):
        get_delay = lambda x, _: x[delay]
    elif not callable(delay):
        get_delay = lambda _, __: delay
    else:
        get_delay = delay

    mementos = []

    for i, (x, y) in enumerate(X_y):

        t = get_moment(i, x)
        d = get_delay(x, y)

        while mementos:

            # Get the oldest answer
            i_old, x_old, y_old, t_expire = mementos[0]

            # If the oldest answer isn't old enough then stop
            if t_expire > t:
                break

            # Reveal the duration and pop the trip from the queue
            yield i_old, x_old, y_old
            del mementos[0]

        yield i, x, None
        queue(mementos, Memento(i, x, y, t + d))

    for memento in mementos:
        yield memento.i, memento.x, memento.y
