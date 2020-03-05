import collections

from . import estimator


class Ensemble(estimator.Estimator, collections.UserList):
    """An ensemble model."""
