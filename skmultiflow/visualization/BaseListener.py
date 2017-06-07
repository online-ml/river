__author__ = 'Guilherme Matsumoto'

from abc import ABCMeta, abstractmethod


class BaseListener(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def onNewTrainStep(self, performancePoint, trainStep):
        pass
