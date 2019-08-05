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

class Initializer:
    """Base class for weights initializers"""
    def initializer(self, shape):
        """This method must return a numpy array with the values of the initialized weights

        Parameters:
            shape (tuple or int): The shape of each new weight (each new entry in the dictionnary)
            random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """When you call an Initializer, it will return a ``collections.defaultdict``
        where each new entry of the defaultdict will be a call for the ``self.initializer``
        method
        """
        init = partial(self.initializer, *args, **kwargs)
        return defaultdict(init)


class Zeros(Initializer):
    """Initializer which return zeros for each new weight

    Example:

        ::

            >>> import numpy as np
            >>> from creme.initializers import Zeros

            >>> init = Zeros()(shape=2)
            >>> print(init['weights'] == np.zeros(2))
            [ True  True]

    """
    def initializer(self, shape):
        return np.zeros(shape)


class Constant(Initializer):
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

    def initializer(self, shape):
        return np.full(shape, self.value)


class Normal(Initializer):
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

    def initializer(self, shape):
        return self.random_state.normal(loc=self.mu, scale=self.sigma, size=shape)
