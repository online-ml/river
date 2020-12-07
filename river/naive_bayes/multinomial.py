import collections
import math

from river.base import tags

from . import base


__all__ = ["MultinomialNB"]


class MultinomialNB(base.BaseNB):
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

    >>> import math
    >>> from river import compose
    >>> from river import feature_extraction
    >>> from river import naive_bayes

    >>> docs = [
    ...     ('Chinese Beijing Chinese', 'yes'),
    ...     ('Chinese Chinese Shanghai', 'yes'),
    ...     ('Chinese Macao', 'yes'),
    ...     ('Tokyo Japan Chinese', 'no')
    ... ]
    >>> model = compose.Pipeline(
    ...     ('tokenize', feature_extraction.BagOfWords(lowercase=False)),
    ...     ('nb', naive_bayes.MultinomialNB(alpha=1))
    ... )
    >>> for sentence, label in docs:
    ...     model = model.learn_one(sentence, label)

    >>> model['nb'].p_class('yes')
    0.75
    >>> model['nb'].p_class('no')
    0.25

    >>> cp = model['nb'].p_feature_given_class

    >>> cp('Chinese', 'yes') == (5 + 1) / (8 + 6)
    True

    >>> cp('Tokyo', 'yes') == (0 + 1) / (8 + 6)
    True
    >>> cp('Japan', 'yes') == (0 + 1) / (8 + 6)
    True

    >>> cp('Chinese', 'no') == (1 + 1) / (3 + 6)
    True

    >>> cp('Tokyo', 'no') == (1 + 1) / (3 + 6)
    True
    >>> cp('Japan', 'no') == (1 + 1) / (3 + 6)
    True

    >>> new_text = 'Chinese Chinese Chinese Tokyo Japan'
    >>> tokens = model['tokenize'].transform_one(new_text)
    >>> jlh = model['nb'].joint_log_likelihood(tokens)
    >>> math.exp(jlh['yes'])
    0.000301
    >>> math.exp(jlh['no'])
    0.000135
    >>> model.predict_one(new_text)
    'yes'

    >>> new_unseen_text = 'Taiwanese Taipei'
    >>> tokens = model['tokenize'].transform_one(new_unseen_text)
    >>> # P(Taiwanese|yes)
    >>> #   = (N_Taiwanese_yes + 1) / (N_yes + N_terms)
    >>> cp('Taiwanese', 'yes') == cp('Taipei', 'yes') == (0 + 1) / (8 + 6)
    True
    >>> cp('Taiwanese', 'no') == cp('Taipei', 'no') == (0 + 1) / (3 + 6)
    True

    >>> # P(yes|Taiwanese Taipei)
    >>> #   ∝ P(Taiwanese|yes) * P(Taipei|yes) * P(yes)
    >>> posterior_yes_given_new_text = (0 + 1) / (8 + 6) * (0 + 1) / (8 + 6) * 0.75
    >>> jlh = model['nb'].joint_log_likelihood(tokens)
    >>> jlh['yes'] == math.log(posterior_yes_given_new_text)
    True

    >>> model.predict_one(new_unseen_text)
    'yes'

    References
    ----------
    [^1]: [Naive Bayes text classification](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
        self.class_totals = collections.Counter()

    def _more_tags(self):
        return {tags.POSITIVE_INPUT}

    def learn_one(self, x, y):
        self.class_counts.update((y,))

        for f, frequency in x.items():
            self.feature_counts[f].update({y: frequency})
            self.class_totals.update({y: frequency})

        return self

    @property
    def classes_(self):
        return list(self.class_counts.keys())

    @property
    def n_terms(self):
        return len(self.feature_counts)

    def p_feature_given_class(self, f, c):
        num = self.feature_counts.get(f, {}).get(c, 0.0) + self.alpha
        den = self.class_totals[c] + self.alpha * self.n_terms
        return num / den

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x):
        return {
            c: math.log(self.p_class(c))
            + sum(
                frequency * math.log(self.p_feature_given_class(f, c)) for f, frequency in x.items()
            )
            for c in self.classes_
        }
