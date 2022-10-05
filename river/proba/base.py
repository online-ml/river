import abc
import typing

from river import base

__all__ = ["BinaryDistribution", "DiscreteDistribution", "ContinuousDistribution"]


class Distribution(base.Base):

    @abc.abstractmethod
    def __call__(self):
        """Probability mass/density function."""

    @abc.abstractmethod
    def sample(self):
        """Sample a random value from the distribution."""

    @property
    @abc.abstractmethod
    def n_samples(self):
        """The number of observed samples."""

class DiscreteDistribution(Distribution):
    """A probability distribution for discrete values."""

    @abc.abstractmethod
    def update(self, x: typing.Hashable):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: typing.Hashable):
        """Reverts the parameters of the distribution for a given observation."""


class BinaryDistribution(Distribution):
    """A probability distribution for discrete values."""

    @abc.abstractmethod
    def update(self, x: bool):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: bool):
        """Reverts the parameters of the distribution for a given observation."""



class ContinuousDistribution(Distribution):
    """A probability distribution for continuous values."""

    @abc.abstractmethod
    def update(self, x: float):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x: float):
        """Reverts the parameters of the distribution for a given observation."""

    @abc.abstractmethod
    def cdf(self, x: float):
        """Cumulative density function, i.e. P(X <= x)."""
