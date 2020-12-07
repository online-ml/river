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

    References
    ----------
    [^1]: [McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D. and Chikkerur, S., 2013, August. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1222-1230)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

    """

    def __init__(self, p: float, seed: int = None):
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)
        self.included = set()

    def transform_one(self, x):

        xt = {}

        for i, xi in x.items():
            if i not in self.included and self.rng.random() < self.p:
                self.included.add(i)
                xt[i] = xi
            xt[i] = xi

        return xt
