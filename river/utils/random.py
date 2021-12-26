import math
import random

__all__ = ["poisson"]


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
