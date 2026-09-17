from __future__ import annotations

import collections
import math
import typing

from river import base, utils

__all__ = ["ALMAClassifier"]


class ALMAClassifier(base.Classifier):
    """Approximate Large Margin Algorithm (ALMA).

    Parameters
    ----------
    p
    alpha
    B
    C

    Attributes
    ----------
    w : collections.defaultdict
        The current weights.
    k : int
        The number of instances seen during training.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.ALMAClassifier()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 82.56%

    References
    ----------
    [^1]: [Gentile, Claudio. "A new approximate maximal margin classification algorithm." Journal of Machine Learning Research 2.Dec (2001): 213-242](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)

    """

    def __init__(
        self, p: float = 2, alpha: float = 0.9, B: float = 1 / 0.9, C: float = 2**0.5
    ) -> None:
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.w: dict[base.typing.FeatureName, float] = collections.defaultdict(float)
        self.k = 1

    def _raw_dot(self, x: dict[base.typing.FeatureName, float]) -> float:
        return utils.math.dot(x, self.w)

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, float], **kwargs: typing.Any
    ) -> dict[base.typing.ClfTarget, float]:
        yp = utils.math.sigmoid(self._raw_dot(x))
        return {False: 1 - yp, True: yp}

    def learn_one(
        self, x: dict[base.typing.FeatureName, typing.Any], y: base.typing.ClfTarget
    ) -> None:
        # Convert 0 to -1
        y = int(y or -1)

        gamma = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)

        if y * self._raw_dot(x) < (1 - self.alpha) * gamma:
            eta = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))

            for i, xi in x.items():
                self.w[i] += eta * y * xi

            norm = utils.math.norm(self.w, order=self.p)

            for i in x:
                self.w[i] /= max(1, norm)

            self.k += 1
