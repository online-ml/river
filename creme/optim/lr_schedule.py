import abc


__all__ = [
    'ConstantLR',
    'InverseScalingLR',
    'OptimalLR'
]


class LRScheduler(abc.ABC):

    @abc.abstractmethod
    def get(self, t: int) -> float:
        pass


class ConstantLR(LRScheduler):
    """Always uses the same learning rate."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get(self, t):
        return self.learning_rate


class InverseScalingLR(LRScheduler):
    """Reduces the learning rate using a power schedule.

    Assuming an iteration counter $t$ starting from 0, the learning rate will be:

    .. math:: \\frac{1}{(t+1)^p}

    where $p$ is a user-defined parameter.

    """

    def __init__(self, learning_rate, power=0.5):
        self.learning_rate = learning_rate
        self.power = power

    def get(self, t):
        return self.learning_rate / (t + 1) ** self.power


class OptimalLR(LRScheduler):
    """Optimal learning schedule as proposed by Léon Bottou.

    References:

        1. `Stochastic Gradient Descent <https://leon.bottou.org/projects/sgd>`_

    """

    def __init__(self, t0=1000, alpha=1e-4):
        self.t0 = t0
        self.alpha = alpha

    def get(self, t):
        return 1. / (self.alpha * (t + self.t0))
