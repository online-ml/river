import collections
import typing

from creme import base

from .predictor import Predictor


class Ensemble(Predictor):
    """An ensemble model.

    Parameters:
        models

    """

    def __init__(self, models: typing.List[Predictor]):
        self.models = list(models)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]
