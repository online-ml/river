"""Weight initialization schemes."""
import numpy as np
from sklearn import utils


__all__ = [
    'Constant',
    'Normal',
    'Zeros'
]


class Constant:
    """Constant initializer which always return the same value.

    Parameters:
        value (float): The constant value

    Example:

        ::

            >>> from creme import optim

            >>> init = optim.initializers.Constant(value=3.14)

            >>> init(shape=1)
            3.14

            >>> init(shape=2)
            array([3.14, 3.14])

    """

    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value) if shape != 1 else self.value


class Zeros(Constant):
    """Initializer which return zeros for each new weight.

    Example:

        ::

            >>> from creme import optim

            >>> init = optim.initializers.Zeros()

            >>> init(shape=1)
            0.0

            >>> init(shape=2)
            array([0., 0.])

    """

    def __init__(self):
        super().__init__(value=0.)


class Normal:
    """Random normal initializer which simulate a normal distribution with specified parameters.

    Parameters:
        mu (float): The mean of the normal distribution
        sigma (float): The standard deviation of the normal distribution

    Example:

        ::

            >>> from creme import optim

            >>> init = optim.initializers.Normal(mu=0, sigma=1, random_state=42)

            >>> init(shape=1)
            0.496714...

            >>> init(shape=2)
            array([-0.1382643 ,  0.64768854])

    """

    def __init__(self, mu=0., sigma=1., random_state=None):
        self.mu = mu
        self.sigma = sigma
        self.random_state = utils.check_random_state(random_state)

    def __call__(self, shape):
        weights = self.random_state.normal(loc=self.mu, scale=self.sigma, size=shape)
        if shape == 1:
            return weights[0]
        return weights
