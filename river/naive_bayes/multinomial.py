import collections
import math

import pandas as pd
import numpy as np

from scipy import sparse

from river.base import tags

from . import base


__all__ = ["MultinomialNB"]


class MultinomialNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    This estimator supports learning with mini-batches. The input vector has to contain positive
    values, such as counts or TF-IDF values.

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

    >>> import pandas as pd

    >>> docs = [
    ...     ('Chinese Beijing Chinese', 'yes'),
    ...     ('Chinese Chinese Shanghai', 'yes'),
    ...     ('Chinese Macao', 'yes'),
    ...     ('Tokyo Japan Chinese', 'no')
    ... ]

    >>> docs = pd.DataFrame(docs, columns = ['docs', 'y'])

    >>> X, y = docs['docs'], docs['y']

    >>> model = compose.Pipeline(
    ...     ('tokenize', feature_extraction.BagOfWords(lowercase=False)),
    ...     ('nb', naive_bayes.MultinomialNB(alpha=1))
    ... )

    >>> model = model.learn_many(X, y)

    >>> model['nb'].feature_counts
    defaultdict(<class 'collections.Counter'>, {'Japan': Counter({'no': 1}), 'Tokyo': Counter({'no': 1}), 'Chinese': Counter({'yes': 5, 'no': 1}), 'Macao': Counter({'yes': 1}), 'Shanghai': Counter({'yes': 1}), 'Beijing': Counter({'yes': 1})})

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

    >>> unseen_data = pd.Series(
    ...    ['Taiwanese Taipei', 'Chinese Shanghai'], name = 'docs', index = ['river', 'rocks'])

    >>> model.predict_proba_many(unseen_data)
                 no       yes
    river  0.446469  0.553531
    rocks  0.118501  0.881499

    >>> model.predict_many(unseen_data)
    river    yes
    rocks    yes
    dtype: object

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
                frequency * math.log(self.p_feature_given_class(f, c))
                for f, frequency in x.items()
            )
            for c in self.classes_
        }

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        y = base.one_hot_encode(y)
        columns, classes = X.columns, y.columns
        y = sparse.csc_matrix(y.sparse.to_coo()).T

        self.class_counts.update(
            {c: count.item() for c, count in zip(classes, y.sum(axis=1))}
        )

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        fc = y @ X

        self.class_totals.update(
            {c: count.item() for c, count in zip(classes, fc.sum(axis=1))}
        )

        # Update feature counts by slicing the sparse matrix per column.
        # Each column correspond to a class.
        for c, i in zip(classes, range(fc.shape[0])):
            if sparse.issparse(fc):
                counts = {
                    c: {
                        columns[f]: count for f, count in zip(fc[i].indices, fc[i].data)
                    }
                }
            else:
                counts = {c: {f: count for f, count in zip(columns, fc[i])}}
            # Transform {classe_i: {token_1: f_1, ... token_n: f_n}} into:
            # [{token_1: {classe_i: f_1}},.. {token_n: {class_i: f_n}}]
            for dict_count in [
                {token: {c: f} for token, f in frequencies.items()}
                for c, frequencies in counts.items()
            ]:

                for f, count in dict_count.items():
                    self.feature_counts[f].update(count)

        return self

    def p_class_many(self) -> pd.DataFrame:
        return base.from_dict(self.class_counts).T[self.class_counts] / sum(
            self.class_counts.values()
        )

    def _feature_log_prob(self, known: list, unknown: list) -> pd.DataFrame:
        if known:
            smooth_fc = (
                base.from_dict(
                    {
                        f: {c: count[c] if c in count else 0 for c in self.class_totals}
                        for f, count in self.feature_counts.items()
                        if f in known
                    }
                )[self.class_totals].T
                + self.alpha
            )
        else:
            smooth_fc = pd.DataFrame(index=self.class_totals)

        if unknown:
            smooth_fc[unknown] = self.alpha

        smooth_fc = np.log(smooth_fc)

        smooth_cc = np.log(
            base.from_dict(self.class_totals) + self.alpha * self.n_terms
        )

        return smooth_fc.subtract(smooth_cc.values, axis="rows")

    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = X.columns
        index = X.index

        known, unknown = [], []
        for f in columns:
            if f in self.feature_counts:
                known.append(f)
            else:
                unknown.append(f)

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        jll = X @ self._feature_log_prob(known, unknown)[columns].T
        jll += np.log(self.p_class_many()).values

        return pd.DataFrame(jll, index=index, columns=self.class_totals.keys())
