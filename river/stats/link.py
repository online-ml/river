from __future__ import annotations

from river import stats


class Link(stats.base.Univariate):
    """A link joins two univariate statistics as a sequence.

    This can be used to pipe the output of one statistic to the input of another. This can be used,
    for instance, to calculate the mean of the variance of a variable. It can also be used to
    compute shifted statistics by piping statistics with an instance of `stats.Shift`.

    Note that a link is not meant to be instantiated via this class definition. Instead, users can
    link statistics together via the `|` operator.

    Parameters
    ----------
    left
    right
        The output from `left`'s `get` method is passed to `right`'s `update` method if `left`'s
        `get` method doesn't produce `None.`

    Examples
    --------

    >>> from river import stats
    >>> stat = stats.Shift(1) | stats.Mean()

    No values have been seen, therefore `get` defaults to the initial value of `stats.Mean`,
    which is 0.

    >>> stat.get()
    0.

    Let us now call `update`.

    >>> stat = stat.update(1)

    The output from `get` will still be 0. The reason is that `stats.Shift` has not enough
    values, and therefore outputs it's default value, which is `None`. The `stats.Mean`
    instance is therefore not updated.

    >>> stat.get()
    0.0

    On the next call to `update`, the `stats.Shift` instance has seen enough values, and
    therefore the mean can be updated. The mean is therefore equal to 1, because that's the
    only value from the past.

    >>> stat = stat.update(3)
    >>> stat.get()
    1.0

    On the subsequent call to update, the mean will be updated with the value 3.

    >>> stat = stat.update(4)
    >>> stat.get()
    2.0

    Note that composing statistics returns a new statistic with it's own name.

    >>> stat.name
    'mean_of_shift_1'

    """

    def __init__(self, left: stats.base.Univariate, right: stats.base.Univariate):
        self.left = left
        self.right = right

    def update(self, x):
        self.left.update(x)
        y = self.left.get()
        if y is not None:
            self.right.update(y)
        return self

    def get(self):
        return self.right.get()

    @property
    def name(self):
        return f"{self.right.name}_of_{self.left.name}"

    def __repr__(self):
        return repr(self.right)
