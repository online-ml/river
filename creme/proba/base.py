import abc


class Distribution(abc.ABC):

    @abc.abstractmethod
    def update(self, x):
        """Updates the parameters of the distribution given a new observation."""

    @abc.abstractmethod
    def mode(self):
        """Returns the value with the largest likelihood."""

    @abc.abstractmethod
    def proba_of(self, x):
        """Probability density or mass function."""
