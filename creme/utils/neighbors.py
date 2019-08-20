import collections


def minkowski_distance(a, b, p):
    """Minkowski distance.

    Parameters:
        a (dict)
        b (dict)
        p (int): Parameter for the Minkowski distance. When ``p=1``, this is equivalent to using
            the Manhattan distance. When ``p=2``, this is equivalent to using the Euclidean
            distance.

    """
    keys = set([*a.keys(), *b.keys()])
    return sum((a.get(k, 0.) - b.get(k, 0.)) ** p for k in keys)


class NearestNeighbors(collections.deque):

    def __init__(self, window_size):
        super().__init__(maxlen=window_size)

    def find(self, x, k):
        """Returns the ``k`` closest neighbors to ``x`` from the current window."""
