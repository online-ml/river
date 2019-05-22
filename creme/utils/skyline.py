import collections


class Skyline(collections.UserList):
    """A skyline is set of points which is not dominated by any other point.

    This implementation uses a block nested loop. Identical observations are all part of the
    skyline if applicable.

    Example:

        ::

            >>> import random
            >>> from creme import utils
            >>> import matplotlib.pyplot as plt

            >>> city_prices = {
            ...     'Bordeaux': 4045,
            ...     'Lyon': 4547,
            ...     'Toulouse': 3278
            ... }

            >>> def random_house():
            ...     city = random.choice(['Bordeaux', 'Lyon', 'Toulouse'])
            ...     size = round(random.gauss(200, 50))
            ...     price = round(random.uniform(0.8, 1.2) * city_prices[city] * size)
            ...     return {'city': city, 'size': size, 'price': price}

            >>> skyline = utils.Skyline(to_min=['price'], to_max=['size'])

            >>> random.seed(42)

            >>> for _ in range(100):
            ...     house = random_house()
            ...     skyline = skyline.update(house)

            >>> print(len(skyline))
            13

            >>> print(skyline[0])
            {'city': 'Toulouse', 'size': 280, 'price': 763202}

            >>> fig, ax = plt.subplots()
            >>> scatter = ax.scatter(
            ...     x=[h['size'] for h in skyline],
            ...     y=[h['price'] for h in skyline]
            ... )
            >>> grid = ax.grid()
            >>> title = ax.set_title('Houses skyline')
            >>> xlabel = ax.set_xlabel('Size')
            >>> ylabel = ax.set_ylabel('Price')

        .. image:: ../_static/skyline_docstring.svg
            :align: center

    References:
        1. `Skyline queries in Python <https://maxhalford.github.io/blog/skyline-queries-in-python/>`_
        2. `The Skyline Operator <https://infolab.usc.edu/csci599/Fall2007/papers/e-1.pdf>`_
        3. `Maintaining Sliding Window Skylineson Data Streams <http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf>`_

    """

    def __init__(self, to_min=None, to_max=None):

        super().__init__()

        self.to_min = [] if to_min is None else to_min
        self.to_max = [] if to_max is None else to_max

        if len(self.to_min) + len(self.to_max) == 0:
            raise ValueError('At least one name has to be specified')

    def _count_diffs(self, a, b):
        n_better = 0
        n_worse = 0

        for f in self.to_min:
            n_better += a[f] < b[f]
            n_worse += a[f] > b[f]

        for f in self.to_max:
            n_better += a[f] > b[f]
            n_worse += a[f] < b[f]

        return n_better, n_worse

    def update(self, x):

        # If the skyline is empty then the first element is part of the skyline
        if not self:
            self.append(x)
            return self

        to_drop = []
        is_dominated = False

        for i, _ in enumerate(self):

            n_better, n_worse = self._count_diffs(x, self[i])

            # x is dominated by one element in the skyline
            if n_worse > 0 and n_better == 0:
                is_dominated = True
                break

            # x dominates one element in the skyline
            if n_better > 0 and n_worse == 0:
                to_drop.append(i)

        if is_dominated:
            return self

        # Remove dominated elements
        if to_drop:
            for i in sorted(to_drop, reverse=True):
                del self[i]

        # Add x to the skyline
        self.append(x)

        return self
