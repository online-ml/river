import collections
import itertools
import math


def make_coeffs(d, m):
    """Precomputes the coefficients of the backshift operator.

    Example:

        >>> make_coeffs(1, 1)
        {0: -1}

        >>> make_coeffs(2, 1)
        {0: -2, 1: 1}

        >>> make_coeffs(3, 1)
        {0: -3, 1: 3, 2: -1}

        >>> make_coeffs(2, 7)
        {6: -2, 13: 1}

    """

    def n_choose_k(n, k):
        f = math.factorial
        return f(n) // f(k) // f(n - k)

    return dict(
        (
            k * m - 1,
            int(math.copysign(1, (k + 1) % 2 - 1)) * n_choose_k(n=d, k=k)
        )
        for k in range(1, d + 1)
    )


class Differencer:
    """A time series differencer.

    Example:

        >>> differencer = Differencer(2); differencer.coeffs
        {0: -2, 1: 1}

        >>> differencer.diff(7, [3, 1])
        2

        >>> differencer.undiff(2, [3, 1])
        7

    References:
        1. `Stationarity and differencing <https://otexts.com/fpp2/stationarity.html>`_

    """

    def __init__(self, d, m=1):

        if d < 0:
            raise ValueError('d must be greater than or equal to 0')

        if m < 1:
            raise ValueError('d must be greater than or equal to 1')

        self.coeffs = make_coeffs(d=d, m=m)

    def __mul__(self, other):
        """Composes two differencers together.

        Example:

            >>> differencer = Differencer(d=3, m=2) * Differencer(d=3, m=1)
            >>> for t, c in sorted(differencer.coeffs.items()):
            ...     print(t, c)
            0 -3
            2 8
            3 -6
            4 -6
            5 8
            7 -3
            8 1

        References:
            1. `Backshift notation <https://otexts.com/fpp2/backshift.html>`_

        """
        coeffs = collections.Counter()
        coeffs.update(self.coeffs)
        coeffs.update(other.coeffs)

        for (t1, c1), (t2, c2) in itertools.product(self.coeffs.items(), other.coeffs.items()):
            coeffs[t1 + t2 + 1] += c1 * c2

        # Remove 0 coefficients
        for t in list(coeffs.keys()):
            if coeffs[t] == 0:
                del coeffs[t]

        differencer = Differencer(0, 0)
        differencer.coeffs = dict(coeffs)
        return differencer

    def diff(self, y, y_previous):
        """Differentiates a value.

            y (float): The value to differentiate.
            y_previous (list of float): The window of previous values. The first element is assumed
                to be the most recent value.

        """
        return y + sum(c * y_previous[t] for t, c in self.coeffs.items())

    def undiff(self, y, y_previous):
        """Undifferentiates a value.

            y (float): The value to differentiate.
            y_previous (list of float): The window of previous values. The first element is assumed
                to be the most recent value.

        """
        return y - sum(c * y_previous[t] for t, c in self.coeffs.items())
