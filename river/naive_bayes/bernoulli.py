import collections
import math

import numpy as np
import pandas as pd

from scipy import sparse

from . import base


__all__ = ["BernoulliNB"]


class BernoulliNB(base.BaseNB):
    """Bernoulli Naive Bayes.

    This estimator supports learning with mini-batches.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).
    true_threshold
        Threshold for binarizing (mapping to booleans) features.

    Attributes
    ----------
    class_counts : collections.Counter
        Number of times each class has been seen.
    feature_counts : collections.defaultdict
        Total frequencies per feature and class.

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
    ...     ('nb', naive_bayes.BernoulliNB(alpha=1))
    ... )

    >>> for sentence, label in docs:
    ...     model = model.learn_one(sentence, label)

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
    0.005184
    >>> math.exp(jlh['no'])
    0.021947
    >>> model.predict_one(new_text)
    'no'

    >>> model.predict_proba_one('test')['yes']
    0.8831539823829913

    Mini-batches:

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
    ...     ('nb', naive_bayes.BernoulliNB(alpha=1))
    ... )

    >>> model = model.learn_many(X, y)

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

    >>> unseen_data = pd.Series(
    ...    ['Taiwanese Taipei', 'Chinese Shanghai'], name = 'docs', index=['river', 'rocks'])

    >>> model.predict_proba_many(unseen_data)
                 no       yes
    river  0.116846  0.883154
    rocks  0.047269  0.952731

    >>> model.predict_many(unseen_data)
    river    yes
    rocks    yes
    dtype: object

    >>> unseen_data = pd.Series(
    ...    ['test'], name = 'docs')

    >>> model.predict_proba_many(unseen_data)
            no       yes
    0  0.116846  0.883154

    >>> model.predict_proba_one('test')
    {'no': 0.116846017617009, 'yes': 0.8831539823829913}


    References
    ----------
    [^1]: [The Bernoulli model](https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html)

    """

    def __init__(self, alpha=1.0, true_threshold=0.0):
        self.alpha = alpha
        self.true_threshold = true_threshold
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)

    def learn_one(self, x, y):
        self.class_counts.update((y,))

        for i, xi in x.items():
            self.feature_counts[i].update({y: xi > self.true_threshold})

        return self

    def p_feature_given_class(self, f: str, c: str) -> float:
        num = self.feature_counts.get(f, {}).get(c, 0.0) + self.alpha
        den = self.class_counts[c] + self.alpha * 2
        return num / den

    def p_class(self, c: str) -> float:
        return self.class_counts[c] / sum(self.class_counts.values())

    def joint_log_likelihood(self, x):
        return {
            c: math.log(self.p_class(c))
            + sum(
                map(
                    math.log,
                    (
                        10e-10 + self.p_feature_given_class(f, c)
                        if f in x and x[f] > self.true_threshold
                        else 10e-10 + (1.0 - self.p_feature_given_class(f, c))
                        for f in self.feature_counts
                    ),
                )
            )
            for c in self.class_counts
        }

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        # One hot encode y and convert it into sparse matrix
        y = base.one_hot_encode(y)
        columns, classes = X.columns, y.columns
        y = sparse.csc_matrix(y.sparse.to_coo()).T

        self.class_counts.update(
            {c: count.item() for c, count in zip(classes, y.sum(axis=1))}
        )

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())
            X.data = X.data > self.true_threshold
        else:
            X = X > self.true_threshold

        fc = y @ X

        for c, i in zip(classes, range(fc.shape[0])):
            counts = {
                c: {columns[f]: count for f, count in zip(fc[i].indices, fc[i].data)}
            }
            # Transform {classe_i: {token_1: f_1, ... token_n: f_n}} into:
            # [{token_1: {classe_i: f_1}},.. {token_n: {class_i: f_n}}]
            for dict_count in [
                {token: {c: f} for token, f in frequencies.items()}
                for c, frequencies in counts.items()
            ]:

                for f, count in dict_count.items():
                    self.feature_counts[f].update(count)

            # HANDLE CASE WHERE X IS NOT A SPARSE DATAFRAME

        return self

    def p_class_many(self):
        return base.from_dict(self.class_counts).T[self.class_counts] / sum(
            self.class_counts.values()
        )

    def feature_log_prob(self, columns):
        smooth_fc = np.log(
            base.from_dict(self.feature_counts)[self.class_counts].T.fillna(0)
            + self.alpha
        )[columns]
        smooth_cc = np.log(base.from_dict(self.class_counts) + self.alpha * 2)

        return smooth_fc.subtract(smooth_cc.values)

    def joint_log_likelihood_many(self, X: pd.DataFrame):
        """Calculate the posterior log probability of the samples X"""
        index = X.index

        unknown = [x for x in X.columns if x not in self.feature_counts]
        missing = [x for x in self.feature_counts if x not in X.columns]

        if unknown:
            X = X.drop(unknown, axis="columns")

        if missing:
            X[missing] = False

        columns = X.columns

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())
            X.data = X.data > self.true_threshold
        else:
            X = X > self.true_threshold

        flp = self.feature_log_prob(columns)

        neg_p = np.log(1 - np.exp(flp))

        jll = X @ (flp - neg_p).T

        pcm = self.p_class_many()

        jll += (np.log(pcm) + neg_p.sum(axis=1).T).values

        return pd.DataFrame(jll, index=index, columns=self.class_counts.keys())
