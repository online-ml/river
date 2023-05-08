from __future__ import annotations

from collections import UserList
from collections.abc import Iterator
from random import Random

from .estimator import Estimator
from .wrapper import Wrapper


class Ensemble(UserList):
    """An ensemble is a model which is composed of a list of models.

    Parameters
    ----------
    models

    """

    def __init__(self, models: Iterator[Estimator]):
        super().__init__(models)

        if len(self) < self._min_number_of_models:
            raise ValueError(
                f"At least {self._min_number_of_models} models are expected, "
                + f"only {len(self)} were passed"
            )

    @property
    def _min_number_of_models(self):
        return 2

    @property
    def models(self):
        return self.data


class WrapperEnsemble(Ensemble, Wrapper):
    """A wrapper ensemble is an ensemble composed of multiple copies of the same model.

    Parameters
    ----------
    model
        The model to copy.
    n_models
        The number of copies to make.
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, model, n_models, seed):
        super().__init__(model.clone() for _ in range(n_models))
        self.model = model
        self.n_models = n_models
        self.seed = seed
        self._rng = Random(seed)

    @property
    def _wrapped_model(self):
        return self.model
