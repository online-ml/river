import abc

__all__ = ["DiscreteDistribution", "ContinuousDistribution"]


class Distribution(abc.ABC):
    @abc.abstractmethod
    def update(self, x):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def revert(self, x):
        """Reverts the parameters of the distribution for a given observation."""

    @property
    @abc.abstractmethod
    def n_samples(self):
        """The number of observed samples."""

    def __repr__(self):
        return str(self)


class DiscreteDistribution(Distribution):
    """A probability distribution for discrete values."""

    @abc.abstractmethod
    def pmf(self, x):
        """Probability mass function."""


class ContinuousDistribution(Distribution):
    """A probability distribution for continuous values."""

    @property
    @abc.abstractmethod
    def mode(self):
        """Most likely value."""

    @abc.abstractmethod
    def pdf(self, x):
        """Probability density function, i.e. P(x <= X < x+dx) / dx."""

    @abc.abstractmethod
    def cdf(self, x):
        """Cumulative density function, i.e. P(X <= x)."""
