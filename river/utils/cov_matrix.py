import collections


class CovMatrix(collections.UserDict):
    def __getitem__(self, key):
        """

        The covariance matrix is symmetric. So instead of raising a KeyError if an (x, y) entry is
        not found, we first check if (y, x) exists.

        """
        x, y = key
        try:
            return super().__getitem__((x, y))
        except KeyError:
            return super().__getitem__((y, x))
