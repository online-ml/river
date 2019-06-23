import abc


class Statistic(abc.ABC):

    @abc.abstractmethod
    def get(self) -> float:
        """Returns the current value of the statistic."""

    def __str__(self):
        return f'{self.__class__.__name__}: {self.get():.6f}'.rstrip('0')

    def __repr__(self):
        return str(self)


class Univariate(Statistic):

    @property
    @abc.abstractmethod
    def name(self):
        """The name may be used for programmatically naming generated features."""

    @abc.abstractmethod
    def update(self, x: float):
        """Updates and returns the called instance."""


class Bivariate(Statistic):

    @abc.abstractmethod
    def update(self, x, y):
        """Updates and returns the called instance."""
