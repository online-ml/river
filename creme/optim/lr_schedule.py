import abc


__all__ = ['ConstantLR', 'InverseScalingLR']


class LRScheduler(abc.ABC):

    @abc.abstractmethod
    def get(self, iteration: int) -> float:
        pass


class ConstantLR(LRScheduler):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get(self, iteration):
        return self.learning_rate


class InverseScalingLR(LRScheduler):

    def __init__(self, learning_rate, power=0.5):
        self.learning_rate = learning_rate
        self.power = power

    def get(self, iteration):
        return self.learning_rate / (iteration + 1) ** self.power
