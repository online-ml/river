"""Learning rate schedulers."""
import abc
import math


__all__ = [
    'Constant',
    'InverseScaling',
    'Optimal'
]


class Scheduler(abc.ABC):

    @abc.abstractmethod
    def get(self, t: int) -> float:
        """Returns the learning rate at a given iteration."""


class Constant(Scheduler):
    """Always uses the same learning rate."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get(self, t):
        return self.learning_rate


class InverseScaling(Scheduler):
    """Reduces the learning rate using a power schedule.

    Assuming an iteration counter $t$ starting from 0, the learning rate will be:

    .. math:: \\frac{1}{(t+1)^p}

    where $p$ is a user-defined parameter.

    """

    def __init__(self, learning_rate, power=0.5):
        self.learning_rate = learning_rate
        self.power = power

    def get(self, t):
        return self.learning_rate / pow(t + 1, self.power)


class Optimal(Scheduler):
    """Optimal learning schedule as proposed by Léon Bottou.

    Parameters:
        loss (optim.losses.Loss)
        alpha (float)

    References:
        1. `Stochastic Gradient Descent <https://leon.bottou.org/projects/sgd>`_
        2. `Stochastic Gradient Descent Tricks <https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf>`_

    """

    def __init__(self, loss, alpha=1e-4):
        self.loss = loss
        self.alpha = alpha

        typw = math.sqrt(1. / math.sqrt(self.alpha))
        initial_eta0 = typw / max(1.0, self.loss.gradient(-typw, 1.0))
        self.t0 = 1. / (initial_eta0 * self.alpha)

    def get(self, t):
        return 1. / (self.alpha * (self.t0 + t))
