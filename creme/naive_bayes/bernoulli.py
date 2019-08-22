import collections
import math

from . import base


__all__ = ['BernoulliNB']


class BernoulliNB(base.BaseNB):
    """Bernoulli Naive Bayes.

    This class inherits ``predict_proba_one`` from ``naive_bayes.BaseNB`` which itself inherits
    ``predict_one`` from `base.MultiClassifier`.

    Parameters:
        alpha (float): Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).
        true_threshold (float): Threshold for binarizing (mapping to booleans) features.

    Attributes:
        class_counts (collections.Counter): Number of times each class has been seen.
        feature_counts (collections.defaultdict): Total frequencies per feature and class.

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
            ...     ('tokenize', feature_extraction.CountVectorizer(lowercase=False)),
            ...     ('nb', naive_bayes.BernoulliNB(alpha=1))
            ... ])
            >>> for sentence, label in docs:
            ...     model = model.fit_one(sentence, label)

            >>> model['nb'].p_class('yes')
            0.75
            >>> model['nb'].p_class('no')
            0.25

            >>> cp = model['nb'].p_feature_given_class

            >>> cp('Chinese', 'yes') == (3 + 1) / (3 + 2)
            True

            >>> cp('Japan', 'yes') == (0 + 1) / (3 + 2)
            True
            >>> cp('Tokyo', 'yes') == (0 + 1) / (3 + 2)
            True

            >>> cp('Beijing', 'yes') == (1 + 1) / (3 + 2)
            True
            >>> cp('Macao', 'yes') == (1 + 1) / (3 + 2)
            True
            >>> cp('Shanghai', 'yes') == (1 + 1) / (3 + 2)
            True

            >>> cp('Chinese', 'no') == (1 + 1) / (1 + 2)
            True

            >>> cp('Japan', 'no') == (1 + 1) / (1 + 2)
            True
            >>> cp('Tokyo', 'no') == (1 + 1) / (1 + 2)
            True

            >>> cp('Beijing', 'no') == (0 + 1) / (1 + 2)
            True
            >>> cp('Macao', 'no') == (0 + 1) / (1 + 2)
            True
            >>> cp('Shanghai', 'no') == (0 + 1) / (1 + 2)
            True

            >>> new_text = 'Chinese Chinese Chinese Tokyo Japan'
            >>> tokens = model['tokenize'].transform_one(new_text)
            >>> jlh = model['nb'].joint_log_likelihood(tokens)
            >>> math.exp(jlh['yes'])
            0.005184...
            >>> math.exp(jlh['no'])
            0.021947...
            >>> model.predict_one(new_text)
            'no'

    References:
        1. `The Bernoulli model <https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html>`_

    """

    def __init__(self, alpha=1., true_threshold=0.):
        self.alpha = alpha
        self.true_threshold = true_threshold
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)

    def fit_one(self, x, y):
        self.class_counts.update((y,))

        for i, xi in x.items():
            self.feature_counts[i].update({y: xi > self.true_threshold})

        return self

    def p_feature_given_class(self, f, c):
        num = self.feature_counts[f][c] + self.alpha
        den = self.class_counts[c] + self.alpha * 2
        return num / den

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x):
        return {
            c: math.log(self.p_class(c)) + sum(map(
                math.log,
                (
                    10e-10 + self.p_feature_given_class(f, c)
                    if f in x and x[f] > self.true_threshold
                    else 10e-10 + (1. - self.p_feature_given_class(f, c))
                    for f in self.feature_counts
                )
            ))
            for c in self.class_counts
        }
