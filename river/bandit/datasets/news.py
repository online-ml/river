from __future__ import annotations

import pathlib

from river import datasets

from .base import BanditDataset


class NewsArticles(datasets.base.RemoteDataset, BanditDataset):
    """News articles bandit dataset.

    This is a personalization dataset. It contains 10000 observations. There are 10 arms, and the
    reward is binary. There are 100 features, which turns this into a contextual bandit problem.

    Examples
    --------

    >>> from river import bandit

    >>> dataset = bandit.datasets.NewsArticles()
    >>> context, arm, reward = next(iter(dataset))

    >>> len(context)
    100

    >>> arm, reward
    (2, False)

    References
    ----------
    [^1]: [Machine Learning for Personalization homework](http://www.cs.columbia.edu/~jebara/6998/hw2.pdf)
    [^2]: [Contextual Bandits Analysis of LinUCB Disjoint Algorithm with Dataset](https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/)

    """

    def __init__(self):
        super().__init__(
            url="http://www.cs.columbia.edu/~jebara/6998/dataset.txt",
            size=2_149_159,
            filename="dataset.txt",
            unpack=False,
            directory=pathlib.Path(__file__).parent,
            n_features=100,
            n_samples=10_000,
        )

    @property
    def arms(self) -> list:
        return list(range(1, 11, 1))

    def _iter(self):
        with open(self.path) as f:
            for x in f:
                arm, reward, *features = x.strip().split(" ")
                arm = int(arm)
                reward = reward == "1"
                features = {i: float(x) for i, x in enumerate(features)}
                yield features, arm, reward
