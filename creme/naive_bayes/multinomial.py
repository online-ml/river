import collections
import functools
import math

from .. import dist

from . import base


__all__ = ['MultinomialNB']


class MultinomialNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    The input vector has to contain positive values, such as counts or TF-IDF values.

    This class inherits ``predict_proba_one`` from ``naive_bayes.BaseNB`` which itself inherits
    ``predict_one`` from ``base.MultiClassifier``.

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
        ...     y_pred = model.fit_one({'text': x}, y)

        >>> model.steps[-1][1].class_dist.pmf('yes')
        0.75
        >>> cp = model.steps[-1][1].p_term_given_class
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
        >>> tokens = model.steps[0][1].transform_one({'text': new_text})
        >>> llh = model.steps[-1][1]._joint_log_likelihood(tokens)
        >>> math.exp(llh['yes'])
        0.0003...
        >>> math.exp(llh['no'])
        0.0001...
        >>> model.predict_one({'text': new_text})
        'yes'

    References:

    1. `Naive Bayes text classification <https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html>`_

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_dist = dist.Multinomial()
        self.term_counts = collections.defaultdict(functools.partial(collections.defaultdict, int))
        self.class_term_counts = collections.defaultdict(int)

    @property
    def n_terms(self):
        return len(self.term_counts)

    def p_term_given_class(self, term, c):
        """Returns P(term | class)."""
        numerator = self.term_counts[term][c] + self.alpha
        denominator = self.class_term_counts[c] + self.alpha * self.n_terms
        return numerator / denominator

    def fit_one(self, x, y):
        y_pred = self.predict_proba_one(x)

        self.class_dist.update(y)

        for term, frequency in x.items():
            self.term_counts[term][y] += frequency
            self.class_term_counts[y] += frequency

        return y_pred

    def _joint_log_likelihood(self, x):
        return {
            c: math.log(self.class_dist.pmf(c)) + sum(
                frequency * math.log(self.p_term_given_class(term, c))
                for term, frequency in x.items()
            )
            for c in self.class_term_counts
        }
