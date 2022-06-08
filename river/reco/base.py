import abc
import numbers
import random
import typing

from river import base

ID = typing.Union[str, int]
Reward = typing.Union[numbers.Number, bool]


__all__ = ["Ranker"]


class Ranker(base.Estimator):
    """Base class for ranking models.

    Parameters
    ----------
    seed
        Random number generation seed. Set this for reproducibility.

    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def is_contextual(self):
        return False

    @abc.abstractmethod
    def learn_one(self, user: ID, item: ID, y: Reward, x: dict = None):
        """Fits a `user`-`item` pair and a real-valued target `y`.

        Parameters
        ----------
        user
            A user ID.
        item
            An item ID.
        y
            Reward feedback from the user for the item. This may be a boolean or a number.
        x
            Optional context to use.

        """

    @abc.abstractmethod
    def predict_one(self, user: ID, item: ID, x: dict = None) -> Reward:
        """Predicts the target value of a set of features `x`.

        Parameters
        ----------
        user
            A user ID.
        item
            An item ID.
        x
            Optional context to use.

        Returns
        -------
        The predicted preference from the user for the item.

        """

    def rank(self, user: ID, items: typing.Set[ID], x: dict = None) -> typing.List[ID]:
        """Rank models by decreasing order of preference for a given user.

        Parameters
        ----------
        user
            A user ID.
        items
            A set of items to rank.
        x
            Optional context to use.

        """
        preferences = {item: self.predict_one(user, item, x) for item in items}
        return sorted(preferences, key=preferences.__getitem__, reverse=True)  # type: ignore[arg-type]

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}
