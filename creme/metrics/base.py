import abc


class Metric(abc.ABC):

    @abc.abstractmethod
    def update(self, y_true: float, y_pred: float):
        pass

    @abc.abstractmethod
    def get(self) -> float:
        pass
