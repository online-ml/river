"""
Initializers module
"""
from collections import defaultdict
from functools import partial
import numpy as np
from sklearn import utils

__all__ = [
    'Zeros',
    'Constant',
    'Normal'
]


class Zeros:
    """Initializer which return zeros for each new weight

    Example:

        ::

            >>> import numpy as np
            >>> from creme.initializers import Zeros

            >>> init = Zeros()(shape=2)
            >>> print(init['weights'] == np.zeros(2))
            [ True  True]

    """
    def __init__(self):
        pass

    def __call__(self, shape):
        return np.zeros(shape) if shape != 1 else 0


class Constant:
    """Constant initializer which always return the same value

    Parameters:
        value (float): The constant value

    Example:

        ::

            >>> import numpy as np
            >>> from creme.initializers import Constant

            >>> init = Constant(3.14)(shape=2)
            >>> print(init['weights'] == np.full(2, 3.14))
            [ True  True]
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value) if shape != 1 else self.value


class Normal:
    """Random normal initializer which simulate a normal distribution with specified parameters

    Parameters:
        mu (float): The mean of the normal distribution
        sigma (float): The standard deviation of the normal distribution

    Example:

        ::

            >>> import numpy as np
            >>> from creme.initializers import Normal

            >>> init = Normal(mu=0., sigma=1., random_state=42)(shape=2)
            >>> np.random.seed(42)
            >>> print(init['weights'] == np.random.normal(0., 1., 2))
            [ True  True]
    """
    def __init__(self, mu=0., sigma=1., random_state=None):
        self.mu = mu
        self.sigma = sigma
        self.random_state = utils.check_random_state(random_state)

    def __call__(self, shape):
        weights = self.random_state.normal(loc=self.mu, scale=self.sigma, size=shape)
        if shape == 1:
            return weights[0]
        else:
            return weights
