import collections


class Skyline(collections.UserList):
    """A skyline is set of points which is not dominated by any other point.

    This implementation uses a block nested loop. Identical observations are all part of the
    skyline if applicable.

    Parameters
    ----------
    minimize
        A list of features for which the values need to be minimized. Can be omitted as
        long as `maximize` is specified.
    maximize
        A list of features for which the values need to be maximized. Can be omitted as
        long as `minimize` is specified.

    Examples
    --------

    Here is an example taken from [this](https://maxhalford.github.io/blog/skyline-queries-in-python) blog post.

    >>> import random
    >>> from river import utils
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

    >>> skyline = utils.Skyline(minimize=['price'], maximize=['size'])

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

    .. image:: ../../docs/img/skyline_docstring.svg
        :align: center

    Here is another example using the kart data from *Mario Kart: Double Dash!!*.

    >>> import collections
    >>> from river import utils

    >>> Kart = collections.namedtuple(
    ...      'Kart',
    ...      'name speed off_road acceleration weight turbo'
    ... )

    >>> karts = [
    ...     Kart('Red Fire', 5, 4, 4, 5, 2),
    ...     Kart('Green Fire', 7, 3, 3, 4, 2),
    ...     Kart('Heart Coach', 4, 6, 6, 5, 2),
    ...     Kart('Bloom Coach', 6, 4, 5, 3, 2),
    ...     Kart('Turbo Yoshi', 4, 5, 6, 6, 2),
    ...     Kart('Turbo Birdo', 6, 4, 4, 7, 2),
    ...     Kart('Goo-Goo Buggy', 1, 9, 9, 2, 3),
    ...     Kart('Rattle Buggy', 2, 9, 8, 2, 3),
    ...     Kart('Toad Kart', 3, 9, 7, 2, 3),
    ...     Kart('Toadette Kart', 1, 9, 9, 2, 3),
    ...     Kart('Koopa Dasher', 2, 8, 8, 3, 3),
    ...     Kart('Para-Wing', 1, 8, 9, 3, 3),
    ...     Kart('DK Jumbo', 8, 2, 2, 8, 1),
    ...     Kart('Barrel Train', 8, 7, 3, 5, 3),
    ...     Kart('Koopa King', 9, 1, 1, 9, 1),
    ...     Kart('Bullet Blaster', 8, 1, 4, 1, 3),
    ...     Kart('Wario Car', 7, 3, 3, 7, 1),
    ...     Kart('Waluigi Racer', 5, 9, 5, 6, 2),
    ...     Kart('Piranha Pipes', 8, 7, 2, 9, 1),
    ...     Kart('Boo Pipes', 2, 9, 8, 9, 1),
    ...     Kart('Parade Kart', 7, 3, 4, 7, 3)
    ... ]

    >>> skyline = utils.Skyline(
    ...     maximize=['speed', 'off_road', 'acceleration', 'turbo'],
    ...     minimize=['weight']
    ... )

    >>> for kart in karts:
    ...     skyline = skyline.update(kart._asdict())

    >>> best_cart_names = [kart['name'] for kart in skyline]
    >>> for name in best_cart_names:
    ...     print(f'- {name}')
    - Green Fire
    - Heart Coach
    - Bloom Coach
    - Goo-Goo Buggy
    - Rattle Buggy
    - Toad Kart
    - Toadette Kart
    - Barrel Train
    - Koopa King
    - Bullet Blaster
    - Waluigi Racer
    - Parade Kart

    >>> for name in sorted(set(kart.name for kart in karts) - set(best_cart_names)):
    ...     print(f'- {name}')
    - Boo Pipes
    - DK Jumbo
    - Koopa Dasher
    - Para-Wing
    - Piranha Pipes
    - Red Fire
    - Turbo Birdo
    - Turbo Yoshi
    - Wario Car

    References
    ----------
    [^1]: [Skyline queries in Python](https://maxhalford.github.io/blog/skyline-queries-in-python)
    [^2]: [Borzsony, S., Kossmann, D. and Stocker, K., 2001, April. The skyline operator. In Proceedings 17th international conference on data engineering (pp. 421-430). IEEE.](https://infolab.usc.edu/csci599/Fall2007/papers/e-1.pdf)
    [^3]: [Tao, Y. and Papadias, D., 2006. Maintaining sliding window skylines on data streams. IEEE Transactions on Knowledge and Data Engineering, 18(3), pp.377-391.](http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf)

    """

    def __init__(self, minimize: list = None, maximize: list = None):

        super().__init__()

        self.minimize = [] if minimize is None else minimize
        self.maximize = [] if maximize is None else maximize

        if len(self.minimize) + len(self.maximize) == 0:
            raise ValueError("At least one name has to be specified")

    def _count_diffs(self, a, b):
        n_better = 0
        n_worse = 0

        for f in self.minimize:
            n_better += a[f] < b[f]
            n_worse += a[f] > b[f]

        for f in self.maximize:
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
