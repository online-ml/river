import abc


class RunningStatistic(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self):
        """The name of the statistic is used for naming automatically generated features."""

    @abc.abstractmethod
    def update(self):
        """Updates the statistic and returns the called instance."""

    @abc.abstractmethod
    def get(self) -> float:
        """Returns the current value of the statistic."""

    def __str__(self):
        return f'{self.__class__.__name__}: {self.get():.6f}'.rstrip('0')
