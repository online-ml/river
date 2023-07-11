from __future__ import annotations

import math
import random

__all__ = ["poisson", "exponential"]


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


def exponential(rate: float = 1.0, rng=random) -> float:
    """Sample a random value from a Poisson distribution.

    Parameters
    ----------
    rate
    rng

    References
    ----------
    [^1]: [Wikipedia article](https://www.wikiwand.com/en/Exponential_distribution#Random_variate_generation)

    """

    u = rng.random()

    # Retrive the λ value from the rate (β): β = 1 / λ
    lmbda = 1.0 / rate
    return -math.log(1 - u) / lmbda
