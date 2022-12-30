import math
import random

__all__ = [
    "poisson",
    "sample_discrete"
]


def poisson(rate: float, rng=random) -> int:
    """Sample a random value from a Poisson distribution.

    Parameters
    ----------
    rate
    rng

    References
    ----------
    [^1] [Wikipedia article](https://www.wikiwand.com/en/Poisson_distribution#/Generating_Poisson-distributed_random_variables)

    """

    L = math.exp(-rate)
    k = 0
    p = 1

    while p > L:
        k += 1
        p *= rng.random()

    return k - 1


def sample_discrete(distribution: list[float]) -> int:
    """Samples according to the given discrete distribution.

    Parameters
    ----------
    distribution : list[float]
        The discrete distribution we want to sample from. This must contain
        non-negative entries that sum to one.

    Returns
    -------
    output : int
        Output sampled in {0, 1, 2, distribution size} according to the given distribution

    Notes
    -----
    It is useless to np.cumsum and np.searchsorted here, since we want a single
    sample for this distribution and it changes at each call. So nothing
    is better here than simple O(n).

    Warning
    -------
    No test is performed here for efficiency: distribution must contain non-negative values that sum to one.
    """
    # Notes
    U = random.uniform(0.0, 1.0)
    cumsum = 0.0
    size = len(distribution)
    for j in range(size):
        cumsum += distribution[j]
        if U <= cumsum:
            return j
    return size - 1
