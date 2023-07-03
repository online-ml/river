from __future__ import annotations

import math

from river import base

from .base import ActiveLearningClassifier


class EntropySampler(ActiveLearningClassifier):
    """Active learning classifier based on entropy measures.

    The entropy sampler selects samples for labeling based on the entropy of the prediction. The
    higher the entropy, the more likely the sample will be selected for labeling. The entropy
    measure is normalized to [0, 1] and then raised to the power of the discount factor.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    discount_factor
        The discount factor to apply to the entropy measure. A value of 1 won't affect the entropy.
        The higher the discount factor, the more the entropy will be discounted, and the less
        likely samples will be selected for labeling. A value of 0 will select all samples for
        labeling. The discount factor is thus a way to control how many samples are selected for
        labeling.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import active
    >>> from river import datasets
    >>> from river import feature_extraction
    >>> from river import linear_model
    >>> from river import metrics

    >>> dataset = datasets.SMSSpam()
    >>> metric = metrics.Accuracy()
    >>> model = (
    ...     feature_extraction.TFIDF(on='body') |
    ...     linear_model.LogisticRegression()
    ... )
    >>> model = active.EntropySampler(model, seed=42)

    >>> n_samples_used = 0
    >>> for x, y in dataset:
    ...     y_pred, ask = model.predict_one(x)
    ...     metric = metric.update(y, y_pred)
    ...     if ask:
    ...         n_samples_used += 1
    ...         model = model.learn_one(x, y)

    >>> metric
    Accuracy: 86.60%

    >>> dataset.n_samples, n_samples_used
    (5574, 1921)

    >>> print(f"{n_samples_used / dataset.n_samples:.2%}")
    34.46%

    """

    def __init__(self, classifier: base.Classifier, discount_factor: float = 3, seed=None):
        super().__init__(classifier, seed=seed)
        self.discount_factor = discount_factor

    def _p(self, y_pred):
        """

        >>> sampler = EntropySampler(classifier=None)
        >>> sampler._p({'a': 0.5, 'b': 0.5})
        1.0

        >>> sampler._p({'a': 0.5, 'b': 0.5, 'c': 0.0, 'd': 0.0})
        1.0

        """
        if not (entropy := -sum(p * math.log2(p) for p in y_pred.values() if p > 0)):
            return 0.0
        # Normalize entropy to [0, 1]. We only consider non-zero probabilities in order to avoid
        # cases where two classes are at 50%, and the rest of the classes are at 0%. In such a
        # case, the entropy would be close to 0, which is not desirable.
        entropy /= math.log2(sum(1 for p in y_pred.values() if p > 0))
        return entropy**self.discount_factor

    def _ask_for_label(self, x, y_pred) -> bool:
        return self._rng.random() < self._p(y_pred)

    @classmethod
    def _unit_test_params(cls):
        from river import tree

        yield {"classifier": tree.HoeffdingTreeClassifier()}
