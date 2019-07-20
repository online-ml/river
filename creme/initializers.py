"""
Initializers module
"""
from collections import defaultdict
from functools import partial
import numpy as np


class Initializer:
    """Base class for weights initializers"""
    def initializer(self, shape):
        """This method must return a numpy array with the values of the initialized weights

        Parameters:
            shape (tuple or int): The shape of each new weight (each new entry in the dictionnary)
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """When you call an initializer, it will return a ```collections.defaultdict```
        where each new entry of the defaultdict will be a call for the ```self.initializer```
        method
        """
        init = partial(self.initializer, *args, **kwargs)
        return defaultdict(init)


class Zeros(Initializer):
    """Initializer which return zeros for each new weight
    """
    def initializer(self, shape):
        return np.zeros(shape)
    

class Constant(Initializer):
    """Constant initializer which always return the same value

    Parameters:
        value (float): The constant value
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
    """
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def initializer(self, shape):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=shape)


if __name__ == "__main__":
    print(Zeros()(shape=2)[0])
    print(Constant(3)(shape=(2, 2))[0])
    print(Normal(0, 1)(shape=5)[0])