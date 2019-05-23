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
    def proba_of(self, x):
        """Probability density or mass function."""

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
