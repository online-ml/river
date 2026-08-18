from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, TypeVar

from river import base, compose, utils

__all__ = ["CalibratedClassifier"]

T = TypeVar("T", bound=base.Classifier | compose.Pipeline)


def _logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


class CalibratedClassifier(base.Wrapper[T], base.Classifier):
    """Calibrates the probability estimates of a classifier using Platt scaling.

    Platt scaling fits a sigmoid curve on top of the raw scores produced by a classifier, so that
    the resulting probabilities better reflect the true likelihood of the positive class. The
    wrapped classifier's predicted probability for the most likely label is taken as a score, and
    a logistic function of the form $\\hat{p} = \\sigma(a \\cdot s + b)$ is fitted to it.

    The two parameters $a$ and $b$ are updated online, one stochastic gradient step per sample,
    minimizing the log-loss with respect to the true labels. Both are initialized to $a = 1$ and
    $b = 0$, so that the calibration starts as a no-op (the identity) and adapts as data flows in.
    The score of each sample is obtained from the wrapped classifier *before* it learns on that
    same sample, giving the calibration an out-of-sample flavour that prevents overfitting.

    This wrapper is meant for binary classifiers. Its `predict_proba_one` output is always keyed
    on `False` and `True`.

    Parameters
    ----------
    classifier
        The binary classifier to wrap.
    lr
        Learning rate used to update $a$ and $b$.

    Attributes
    ----------
    a
        Slope of the fitted sigmoid.
    b
        Intercept of the fitted sigmoid.

    Examples
    --------

    >>> from river import calibration
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.3301120464388312

    The calibrated version spreads the probabilities further apart when the model is confident and
    pulls them back towards 0.5 when it is not, which typically yields a lower log-loss:

    >>> wrapped = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    >>> model = calibration.CalibratedClassifier(wrapped)

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.3181

    References
    ----------
    [^1]: [Platt, J.C., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3), pp.61-74](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bias.pdf)

    """

    def __init__(self, classifier: T, lr: float = 0.1):
        self.classifier = classifier
        self.lr = lr
        self.a = 1.0
        self.b = 0.0

    @property
    def _wrapped_model(self) -> T:
        return self.classifier

    @property
    def _multiclass(self) -> bool:
        return False

    def _score_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> tuple[base.typing.ClfTarget, float]:
        y_pred = self.classifier.predict_proba_one(x, **kwargs)
        label, p = max(y_pred.items(), key=lambda kv: kv[1])
        return label, _logit(p)

    def learn_one(
        self, x: dict[base.typing.FeatureName, Any], y: base.typing.ClfTarget, **kwargs: Any
    ) -> None:
        label, s = self._score_one(x, **kwargs)
        y_num = float(y == label)

        p = utils.math.sigmoid(self.a * s + self.b)
        self.a -= self.lr * (p - y_num) * s
        self.b -= self.lr * (p - y_num)

        self.classifier.learn_one(x, y, **kwargs)

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        label, s = self._score_one(x, **kwargs)
        p = utils.math.sigmoid(self.a * s + self.b)
        if label is True:
            return {False: 1 - p, True: p}
        return {False: p, True: 1 - p}

    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, compose.Pipeline]]:
        from river import linear_model, preprocessing

        yield {"classifier": (preprocessing.StandardScaler() | linear_model.LogisticRegression())}
