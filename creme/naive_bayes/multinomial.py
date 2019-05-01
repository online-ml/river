import collections
import math

from .. import dist

from . import base


__all__ = ['MultinomialNB']


class MultinomialNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    The input vector has to contain positive values, such as counts or TF-IDF values.

    This class inherits ``predict_proba_one`` from ``naive_bayes.BaseNB`` which itself inherits
    ``predict_one`` from `base.MultiClassifier`.

    Parameters:
        alpha (float): Smoothing parameter used for avoiding zero probabilities.

    Attributes:
        n (int): Number of seen observations.
        class_counts (collections.defaultdict): Number of times each class has been seen.
        term_counts (collections.defaultdict): Number of times each term has been seen.
        class_term_counts (collections.defaultdict): Number of times each term has been seen per
            class.

    Example:

        ::

            >>> import math
            >>> from creme import compose
            >>> from creme import feature_extraction
            >>> from creme import naive_bayes

            >>> docs = [
            ...     ('Chinese Beijing Chinese', 'yes'),
            ...     ('Chinese Chinese Shanghai', 'yes'),
            ...     ('Chinese Macao', 'yes'),
            ...     ('Tokyo Japan Chinese', 'no')
            ... ]
            >>> model = compose.Pipeline([
            ...     ('tokenize', feature_extraction.CountVectorizer(on='text', lowercase=False)),
            ...     ('nb', naive_bayes.MultinomialNB(alpha=1))
            ... ])
            >>> for x, y in docs:
            ...     model = model.fit_one({'text': x}, y)

            >>> model['nb'].p_class('yes')
            0.75
            >>> cp = model['nb'].p_feature_given_class
            >>> cp('Chinese', 'yes') ==  3 / 7
            True
            >>> cp('Tokyo', 'yes') ==  1 / 14
            True
            >>> cp('Japan', 'yes') ==  1 / 14
            True
            >>> cp('Chinese', 'no') ==  2 / 9
            True
            >>> cp('Tokyo', 'no') ==  2 / 9
            True
            >>> cp('Japan', 'no') ==  2 / 9
            True

            >>> new_text = 'Chinese Chinese Chinese Tokyo Japan'
            >>> tokens = model['tokenize'].transform_one({'text': new_text})
            >>> jlh = model['nb']._joint_log_likelihood(tokens)
            >>> math.exp(jlh['yes'])
            0.0003...
            >>> math.exp(jlh['no'])
            0.0001...
            >>> model.predict_one({'text': new_text})
            'yes'

    References:

        1. `Naive Bayes text classification <https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html>`_

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_dist = dist.Multinomial()
        self.feature_frequencies = collections.defaultdict(collections.Counter)
        self.class_total_frequencies = collections.Counter()

    def fit_one(self, x, y):
        self.class_dist.update(y)

        for feature, frequency in x.items():
            self.feature_frequencies[feature].update({y: frequency})
            self.class_total_frequencies.update({y: frequency})

        return self

    @property
    def classes_(self):
        return list(self.class_total_frequencies.keys())

    @property
    def n_terms(self):
        return len(self.feature_frequencies)

    def p_feature_given_class(self, f, c):
        num = self.feature_frequencies[f][c] + self.alpha
        den = self.class_total_frequencies[c] + self.alpha * self.n_terms
        return num / den

    def p_class(self, c):
        return self.class_dist.pmf(c)

    def _joint_log_likelihood(self, x):
        return {
            c: math.log(self.p_class(c)) + sum(
                frequency * math.log(self.p_feature_given_class(feature, c))
                for feature, frequency in x.items()
            )
            for c in self.classes_
        }
