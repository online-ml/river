from collections import UserList
from typing import List

from .estimator import Estimator


class EnsembleMixin(UserList):
    """An ensemble is a model which is composed of a list of models.

    Parameters
    ----------
    models

    """

    def __init__(self, *models: List[Estimator]):
        super().__init__(models)

    @property
    def models(self):
        return self.data
