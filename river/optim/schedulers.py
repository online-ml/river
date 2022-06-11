"""Learning rate schedulers."""
from __future__ import annotations

import math

from river import optim

__all__ = ["Constant", "InverseScaling", "Optimal"]


class Constant(optim.base.Scheduler):
    """Always uses the same learning rate.

    Parameters
    ----------
    learning_rate

    """

    def __init__(self, learning_rate: int | float):
        self.learning_rate = learning_rate

    @property
    def _mutable_attributes(self):
        return {"learning_rate"}

    def get(self, t):
        return self.learning_rate


class InverseScaling(optim.base.Scheduler):
    r"""Reduces the learning rate using a power schedule.

    Assuming an initial learning rate $\eta$, the learning rate at step $t$ is:

    $$\\frac{eta}{(t + 1) ^ p}$$

    where $p$ is a user-defined parameter.

    Parameters
    ----------
    learning_rate
    power

    """

    def __init__(self, learning_rate: float, power=0.5):
        self.learning_rate = learning_rate
        self.power = power

    def get(self, t):
        return self.learning_rate / pow(t + 1, self.power)


class Optimal(optim.base.Scheduler):
    """Optimal learning schedule as proposed by Léon Bottou.

    Parameters
    ----------
    loss
    alpha

    References
    ----------
    [^1]: [Bottou, L., 2012. Stochastic gradient descent tricks. In Neural networks: Tricks of the trade (pp. 421-436). Springer, Berlin, Heidelberg.](https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf)

    """

    def __init__(self, loss: optim.losses.Loss, alpha=1e-4):
        self.loss = loss
        self.alpha = alpha

        typw = math.sqrt(1.0 / math.sqrt(self.alpha))
        initial_eta0 = typw / max(1.0, self.loss.gradient(True, -typw))
        self.t0 = 1.0 / (initial_eta0 * self.alpha)

    def get(self, t):
        return 1.0 / (self.alpha * (self.t0 + t))
