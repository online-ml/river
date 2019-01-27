import abc


__all__ = ['ConstantLR', 'LinearDecreaseLR']


class LRScheduler(abc.ABC):

    def __init__(self, init_lr):
        self.init_lr = init_lr

    @abc.abstractmethod
    def get(self, t: int) -> float:
        pass


class ConstantLR(LRScheduler):

    def get(self, t):
        return self.init_lr


class LinearDecreaseLR(LRScheduler):
    """Implements the decreasing LR schedule proposed by Léon Bottou.

    See somewhere around line 64 of [Léon Bottou's code](https://leon.bottou.org/git/?p=sgd.git;a=blob;f=README.txt)
    for some details.

    """

    def __init__(self, init_lr, strengh):
        super().__init__(init_lr)
        self.strengh = strengh

    def get(self, t):
        return self.init_lr / (1 + t) ** self.strengh
