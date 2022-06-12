def test_with_counter():
    """
    >>> from river import utils
    >>> import collections
    >>> collections.Counter.revert = collections.Counter.subtract

    >>> counter = utils.Rolling(collections.Counter(), window_size=3)

    >>> for i in range(5):
    ...     counter = counter.update([i])

    >>> counter
    Counter({2: 1, 3: 1, 4: 1, 0: 0, 1: 0})

    >>> counter.most_common(3)
    [(2, 1), (3, 1), (4, 1)]

    >>> counter[4]
    1

    """
