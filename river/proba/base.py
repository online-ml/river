import abc
import random
import typing

from river import base

__all__ = ["BinaryDistribution", "DiscreteDistribution", "ContinuousDistribution"]


class Distribution(base.Base):
    """General distribution.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self._rng = random.Random(seed)

    @abc.abstractmethod
    def __call__(self, x: typing.Any) -> float:
        """Probability mass/density function."""

    @abc.abstractmethod
    def sample(self):
        """Sample a random value from the distribution."""

    @property
    @abc.abstractmethod
    def n_samples(self):
        """The number of observed samples."""

    @property
    @abc.abstractmethod
    def mode(self):
        """The most likely value in the distribution."""

    def __gt__(self, other):
        return self.mode > other.mode


class DiscreteDistribution(Distribution):
    """A probability distribution for discrete values.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    """

    @abc.abstractmethod
    def update(self, x: typing.Hashable):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: typing.Hashable):
        """Reverts the parameters of the distribution for a given observation."""


class BinaryDistribution(Distribution):
    """A probability distribution for discrete values.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    """

    @abc.abstractmethod
    def update(self, x: bool):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: bool):
        """Reverts the parameters of the distribution for a given observation."""


class ContinuousDistribution(Distribution):
    """A probability distribution for continuous values.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    """

    @abc.abstractmethod
    def update(self, x: float):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: float):
        """Reverts the parameters of the distribution for a given observation."""

    @abc.abstractmethod
    def cdf(self, x: float):
        """Cumulative density function, i.e. P(X <= x)."""
