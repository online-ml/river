import collections
import math

import pandas as pd

from river.base import tags

from . import base


__all__ = ["ComplementNB"]


class ComplementNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    The input vector has to contain positive values, such as counts or TF-IDF values.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    class_dist : proba.Multinomial
        Class prior probability distribution.
    feature_counts : collections.defaultdict
        Total frequencies per feature and class.
    class_totals : collections.Counter
        Total frequencies per class.

    Examples
    --------

    >>> from river import feature_extraction
    >>> from river import naive_bayes

    >>> sentences = [
    ...     ('food food meat brain', 'health'),
    ...     ('food meat ' + 'kitchen ' * 9 + 'job' * 5, 'butcher'),
    ...     ('food food meat job', 'health')
    ... ]

    >>> model = feature_extraction.BagOfWords() | ('nb', naive_bayes.ComplementNB)

    >>> for sentence, label in sentences:
    ...     model = model.learn_one(sentence, label)

    #>>> model['nb'].feature_counts
    #defaultdict(<class 'collections.Counter'>, {'food': Counter({'health': 4, 'butcher': 1}), 'meat': Counter({'health': 2, 'butcher': 1}), 'brain': Counter({'health': 1}), 'kitchen': Counter({'butcher': 9}), 'jobjobjobjobjob': Counter({'butcher': 1}), 'job': Counter({'health': 1})})

    #>>> model['nb'].class_totals
    #Counter({'butcher': 12, 'health': 8})

    >>> model['nb'].feature_totals
    Counter({'kitchen': 9, 'food': 5, 'meat': 3, 'brain': 1, 'jobjobjobjobjob': 1, 'job': 1})

    #>>> model['nb'].class_counts
    Counter({'health': 2, 'butcher': 1})

    >>> model['nb'].p_class('health') == 2 / 3
    True
    >>> model['nb'].p_class('butcher') == 1 / 3
    True

    >>> model.predict_proba_one('food job meat')
    {'health': 0.779191, 'butcher': 0.220808}

    >>> import pandas as pd

    >>> sentences = [
    ...     ('food food meat brain', 'health'),
    ...     ('food meat ' + 'kitchen ' * 9 + 'job' * 5, 'butcher'),
    ...     ('food food meat job', 'health')
    ... ]

    >>> docs = pd.DataFrame(sentences, columns = ['docs', 'y'])

    >>> X, y = docs['docs'], docs['y']

    >>> model = feature_extraction.BagOfWords() | ('nb', naive_bayes.ComplementNB)

    >>> model = model.learn_many(X, y)

    >>> model['nb'].p_class('health') == 2 / 3
    True
    >>> model['nb'].p_class('butcher') == 1 / 3
    True

    >>> model.predict_proba_one('food job meat')
    {'health': 0.779191, 'butcher': 0.220808}

    References
    ----------
    [^1]: [Rennie, J.D., Shih, L., Teevan, J. and Karger, D.R., 2003. Tackling the poor assumptions of naive bayes text classifiers. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 616-623)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
    [^2]: [StackExchange discussion](https://stats.stackexchange.com/questions/126009/complement-naive-bayes)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
        self.feature_totals = collections.Counter()
        self.class_totals = collections.Counter()

    def _more_tags(self):
        return {tags.POSITIVE_INPUT}

    def learn_one(self, x, y):
        self.class_counts.update((y,))

        for f, frequency in x.items():
            self.feature_counts[f].update({y: frequency})
            self.feature_totals.update({f: frequency})
            self.class_totals.update({y: frequency})

        return self

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        agg = X.groupby(y).sum()

        self.feature_counts.update((agg.T).to_dict(orient="index"))
        self.feature_totals.update(X.sum(axis="rows").to_dict())
        self.class_counts.update(y.value_counts().to_dict())
        self.class_totals.update(agg.sum(axis="columns").to_dict())

        return self

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x):
        return {
            c: sum(
                (
                    frequency
                    * -math.log(
                        (
                            self.feature_totals[f]
                            - self.feature_counts.get(f, {}).get(c, 0.0)
                            + 1
                        )
                        / (self.class_totals[c] + 1 * len(self.feature_counts))
                    )
                    for f, frequency in x.items()
                )
            )
            for c in self.class_counts
        }
