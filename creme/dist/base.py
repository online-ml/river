import abc
import copy


class Distribution(abc.ABC):

    @abc.abstractmethod
    def update(self, x):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def mode(self):
        """Returns the value with the largest likelihood."""

    @abc.abstractmethod
    def __iadd__(self, other):
        """Merges with another distribution or a constant, in-place."""

    @abc.abstractmethod
    def __imul__(self, constant):
        """Scales a distribution by a constant, in-place."""

    def __add__(self, other):
        clone = copy.copy(self)
        clone += other
        return clone

    def __mul__(self, constant):
        clone = copy.copy(self)
        clone *= constant
        return clone

    def __rmul__(self, constant):
        return self * constant


class ContinuousDistribution(Distribution):

    @abc.abstractmethod
    def pdf(self, x):
        """Probability density function."""


class DiscreteDistribution(Distribution):

    @abc.abstractmethod
    def pmf(self, x):
        """Probability mass function."""
