import abc
import numbers
import random
import typing

import numpy as np

from river import base

ID = typing.Union[str, int]
Reward = typing.Union[numbers.Number, bool]


class Recommender(base.Estimator):
    """Base class for recommendation models.

    Parameters
    ----------
    seed
        Random number generation seed. Set this for reproducibility.

    """

    def __init__(self, seed: int = None):
        self.seed = seed
        self._rng = random.Random(seed)
        self._numpy_rng = np.random.RandomState(seed)
        self._items = set()

    @property
    def is_contextual(self):
        return False

    def learn_one(self, x, y: Reward):
        x = x.copy()
        user = x.pop("user")
        item = x.pop("item")
        self._items.add(item)
        return self._learn_user_item(user, item, context=x, reward=y)

    def predict_one(self, x) -> float:
        x = x.copy()
        user = x.pop("user")
        item = x.pop("item")
        self._items.add(item)
        return self._predict_user_item(user, item, context=x)

    @abc.abstractmethod
    def _learn_user_item(
        self, user: ID, item: ID, context: typing.Optional[dict], reward: Reward
    ) -> "Recommender":
        """Fits a `user`-`item` pair and a real-valued target `y`.

        Parameters
        ----------
        user
            A user ID.
        item
            An item ID.
        context
            Side information.
        reward
            Feedback from the user for this item.

        """

    @abc.abstractmethod
    def _predict_user_item(
        self, user: ID, item: ID, context: typing.Optional[dict]
    ) -> float:
        """Predicts the target value of a set of features `x`.

        Parameters
        ----------
        user
            A user ID.
        item
            An item ID.
        context
            Side information.

        Returns
        -------
        The predicted rating.

        """

    def recommend(
        self,
        user: ID,
        k=1,
        context: typing.Optional[dict] = None,
        items: typing.Optional[typing.Set[ID]] = None,
        strategy="best",
    ) -> typing.List[ID]:
        """Recommend k items to a user.

        Parameters
        ----------
        user
            A user ID.
        k
            The number of items to recommend.
        context
            Side information.
        items
            An optional set of items that should be considered. Every seen item will be considered
            if this isn't specified.
        strategy
            The strategy used to select which items to recommend once they've been scored.

        """

        items = list(items or self._items)
        if not items:
            return []

        # Evaluate the preference of each user towards each time given the context
        preferences = [
            self._predict_user_item(user, item, context=context) for item in items
        ]

        # Apply the selection strategy
        if strategy == "best":
            return [item for _, item in sorted(zip(preferences, items), reverse=True)][
                :k
            ]

        raise ValueError(
            f"{strategy} is not a valid value for strategy, must be one of: best"
        )

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}
