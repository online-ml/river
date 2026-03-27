"""Dummy estimators.

This module is here for testing purposes, as well as providing baseline performances.

"""

from __future__ import annotations

import collections
from collections.abc import Iterator

from river import base, stats

__all__ = ["NoChangeClassifier", "PriorClassifier", "StatisticRegressor"]


class NoChangeClassifier(base.Classifier):
    """Dummy classifier which returns the last class seen.

    The predict_one method will output the last class seen whilst predict_proba_one will
    return 1 for the last class seen and 0 for the others.

    Attributes
    ----------
    last_class
        The last class seen.
    classes
        The set of classes seen.

    Examples
    --------

    Taken from example 2.1 from
    [this page](https://www.cms.waikato.ac.nz/~abifet/book/chapter_2.html).

    >>> import pprint
    >>> from river import dummy

    >>> sentences = [
    ...     ('glad happy glad', '+'),
    ...     ('glad glad joyful', '+'),
    ...     ('glad pleasant', '+'),
    ...     ('miserable sad glad', '−')
    ... ]

    >>> model = dummy.NoChangeClassifier()

    >>> for sentence, label in sentences:
    ...     model.learn_one(sentence, label)

    >>> new_sentence = 'glad sad miserable pleasant glad'
    >>> model.predict_one(new_sentence)
    '−'

    >>> pprint.pprint(model.predict_proba_one(new_sentence))
    {'+': 0.0, '−': 1}

    """

    def __init__(self) -> None:
        self.last_class: base.typing.ClfTarget | None = None
        self.classes: set[base.typing.ClfTarget] = set()

    @property
    def _multiclass(self) -> bool:
        return True

    def learn_one(self, x: object, y: base.typing.ClfTarget) -> None:
        self.last_class = y
        self.classes.add(y)

    def predict_one(self, x: object, **kwargs: object) -> base.typing.ClfTarget | None:
        return self.last_class

    def predict_proba_one(self, x: object, **kwargs: object) -> dict[base.typing.ClfTarget, float]:
        probas = {c: 0.0 for c in self.classes}
        if self.last_class is not None:
            probas[self.last_class] = 1
        return probas


class PriorClassifier(base.Classifier):
    """Dummy classifier which uses the prior distribution.

    The `predict_one` method will output the most common class whilst `predict_proba_one` will
    return the normalized class counts.

    Attributes
    ----------
    counts : collections.Counter
        Class counts.
    n : int
        Total number of seen instances.

    Examples
    --------

    Taken from example 2.1 from
    [this page](https://www.cms.waikato.ac.nz/~abifet/book/chapter_2.html)

    >>> from river import dummy

    >>> sentences = [
    ...     ('glad happy glad', '+'),
    ...     ('glad glad joyful', '+'),
    ...     ('glad pleasant', '+'),
    ...     ('miserable sad glad', '−')
    ... ]

    >>> model = dummy.PriorClassifier()

    >>> for sentence, label in sentences:
    ...     model.learn_one(sentence, label)

    >>> new_sentence = 'glad sad miserable pleasant glad'
    >>> model.predict_one(new_sentence)
    '+'
    >>> model.predict_proba_one(new_sentence)
    {'+': 0.75, '−': 0.25}

    References
    ----------
    [^1]: [Krichevsky–Trofimov estimator](https://www.wikiwand.com/en/Krichevsky%E2%80%93Trofimov_estimator)

    """

    def __init__(self) -> None:
        self.counts: collections.Counter[base.typing.ClfTarget] = collections.Counter()
        self.n = 0

    @property
    def _multiclass(self) -> bool:
        return True

    def learn_one(self, x: object, y: base.typing.ClfTarget) -> None:
        self.counts.update([y])
        self.n += 1

    def predict_proba_one(self, x: object, **kwargs: object) -> dict[base.typing.ClfTarget, float]:
        return {label: count / self.n for label, count in self.counts.items()}


class StatisticRegressor(base.Regressor):
    """Dummy regressor that uses a univariate statistic to make predictions.

    Parameters
    ----------
    statistic

    Examples
    --------

    >>> from pprint import pprint
    >>> from river import dummy
    >>> from river import stats

    >>> sentences = [
    ...     ('glad happy glad', 3),
    ...     ('glad glad joyful', 3),
    ...     ('glad pleasant', 2),
    ...     ('miserable sad glad', -3)
    ... ]

    >>> model = dummy.StatisticRegressor(stats.Mean())

    >>> for sentence, score in sentences:
    ...     model.learn_one(sentence, score)

    >>> new_sentence = 'glad sad miserable pleasant glad'
    >>> model.predict_one(new_sentence)
    1.25

    """

    def __init__(self, statistic: stats.base.Univariate):
        self.statistic = statistic

    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, object]]:
        yield {"statistic": stats.Mean()}

    def learn_one(self, x: object, y: base.typing.RegTarget) -> None:
        self.statistic.update(y)  # type: ignore[arg-type]

    def predict_one(self, x: object) -> base.typing.RegTarget | None:  # type: ignore[override]
        return self.statistic.get()
