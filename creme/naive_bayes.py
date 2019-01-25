import collections
import math

from . import base


class MultinomialNB(base.MultiClassifier):
    """

    The input vector has to contain positive values, such as counts or TF-IDF values.

    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

    Example
    -------

        #!python
        >>> import math
        >>> import creme.feature_extraction
        >>> import creme.naive_bayes
        >>> import creme.pipeline

        >>> docs = [
        ...     ('Chinese Beijing Chinese', 'yes'),
        ...     ('Chinese Chinese Shanghai', 'yes'),
        ...     ('Chinese Macao', 'yes'),
        ...     ('Tokyo Japan Chinese', 'no')
        ... ]
        >>> model = creme.pipeline.Pipeline([
        ...     ('tokenize', creme.feature_extraction.CountVectorizer(on='text', lowercase=False)),
        ...     ('nb', creme.naive_bayes.MultinomialNB(alpha=1))
        ... ])
        >>> for x, y in docs:
        ...     y_pred = model.fit_one({'text': x}, y)

        >>> model.steps[-1][1].p_class('yes')
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
        >>> llh = model.steps[-1][1].calc_log_likelihoods(tokens)
        >>> math.exp(llh['yes'])
        0.0003...
        >>> math.exp(llh['no'])
        0.0001...
        >>> model.predict_one({'text': new_text})
        'yes'

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.n = 0
        self.class_counts = collections.defaultdict(lambda: 0)
        self.term_counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        self.class_term_counts = collections.defaultdict(lambda: 0)

    @property
    def n_terms(self):
        return len(self.term_counts)

    def p_class(self, c):
        """Return P(class)."""
        return self.class_counts.get(c, 0) / self.n

    def p_term_given_class(self, term, c):
        """Return P(term | class)."""
        numerator = self.term_counts[term][c] + self.alpha
        denominator = self.class_term_counts[c] + self.alpha * self.n_terms
        return numerator / denominator

    def fit_one(self, x, y):
        y_pred = self.predict_proba_one(x)
        self.n += 1
        self.class_counts[y] += 1
        for term, frequency in x.items():
            self.term_counts[term][y] += frequency
            self.class_term_counts[y] += frequency
        return y_pred

    def calc_log_likelihoods(self, x):
        return {
            c: math.log(self.p_class(c)) + sum(
                frequency * math.log(self.p_term_given_class(term, c))
                for term, frequency in x.items()
            )
            for c in self.class_counts
        }

    def predict_proba_one(self, x):
        llh = self.calc_log_likelihoods(x)
        total = sum(llh.values())
        return {c: likelihood / total for c, likelihood in llh.items()}

    def predict_one(self, x):
        llh = self.calc_log_likelihoods(x)
        return max(llh, key=llh.get)
