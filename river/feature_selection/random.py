from __future__ import annotations

import random

from river import base

__all__ = ["PoissonInclusion"]


class PoissonInclusion(base.Transformer):
    """Randomly selects features with an inclusion trial.

    When a new feature is encountered, it is selected with probability `p`. The number of times a
    feature needs to beseen before it is added to the model follows a geometric distribution with
    expected value `1 / p`. This feature selection method is meant to be used when you have a
    very large amount of sparse features.

    Parameters
    ----------
    p
        Probability of including a feature the first time it is encountered.
    seed
        Random seed value used for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import feature_selection
    >>> from river import stream

    >>> selector = feature_selection.PoissonInclusion(p=0.1, seed=42)

    >>> dataset = iter(datasets.TrumpApproval())

    >>> feature_names = next(dataset)[0].keys()
    >>> n = 0

    >>> while True:
    ...     x, y = next(dataset)
    ...     xt = selector.transform_one(x)
    ...     if xt.keys() == feature_names:
    ...         break
    ...     n += 1

    >>> n
    12

    References
    ----------
    [^1]: [McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D. and Chikkerur, S., 2013, August. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222-1230)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

    """

    def __init__(self, p: float, seed: int | None = None):
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)
        self.included: set[base.typing.FeatureName] = set()

    def transform_one(self, x):
        xt = {}

        for i, xi in x.items():
            if i in self.included:
                xt[i] = xi
            elif self.rng.random() < self.p:
                self.included.add(i)
                xt[i] = xi

        return xt
