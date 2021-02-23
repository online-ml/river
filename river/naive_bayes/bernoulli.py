import collections
import math

import numpy as np
import pandas as pd
from scipy import sparse

from . import base

__all__ = ["BernoulliNB"]


class BernoulliNB(base.BaseNB):
    """Bernoulli Naive Bayes.

    Bernoulli Naive Bayes model learns from occurrences between features such as word
    counts and discrete classes. The input vector must contain positive values, such as
    counts or TF-IDF values.

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

    You can train the model and make predictions in mini-batch mode using the class methods `learn_many` and `predict_many`.

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
    ...    ['Taiwanese Taipei', 'Chinese Shanghai'],
    ...    name = 'docs',
    ...    index=['river', 'rocks']
    ... )

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

    >>> bag = feature_extraction.BagOfWords(lowercase=False)

    >>> model = naive_bayes.BernoulliNB(alpha=1)

    >>> X, y = docs['docs'], docs['y']

    >>> X = bag.transform_many(X)

    >>> X = pd.DataFrame(X.values, columns=X.columns, dtype=int)

    >>> model = model.learn_many(X, y)

    >>> unseen_data = bag.transform_many(unseen_data)

    >>> unseen_data = pd.DataFrame(unseen_data.values, columns=unseen_data.columns, index=['river'])

    >>> model.predict_proba_many(unseen_data)
            no       yes
    river  0.116846  0.883154


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
        """Updates the model with a single observation.

        Parameters
        ----------
        x
            Dictionary of term frequencies.
        y
            Target class.

        Returns
        --------
        self

        """
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

    def p_class_many(self) -> pd.DataFrame:
        return base.from_dict(self.class_counts).T[self.class_counts] / sum(
            self.class_counts.values()
        )

    def joint_log_likelihood(self, x):
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        x
            Dictionary of term frequencies.

        Returns
        --------
        Mapping between classes and joint log likelihood.

        """
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
        """Updates the model with a term-frequency or TF-IDF pandas dataframe.

        Parameters
        ----------
        X
            Term-frequency or TF-IDF pandas dataframe.
        y
            Target classes.

        Returns
        --------
        self

        """
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

        # Update the defaultdict feature counts by slicing the sparse matrix per row.
        # Each row of y @ X correspond to a class and each column to a token.
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

    def _feature_log_prob(self, columns: list) -> pd.DataFrame:
        """Compute log probabilities of input features.

        Parameters
        ----------
        columns
            List of input features.

        Returns
        --------
            Log probabilities of input features.

        """
        smooth_fc = np.log(
            base.from_dict(self.feature_counts)[self.class_counts].T.fillna(0)
            + self.alpha
        )[columns]

        smooth_cc = np.log(base.from_dict(self.class_counts) + self.alpha * 2)

        return smooth_fc.subtract(smooth_cc.values)

    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes the joint log likelihood of input features.

        Parameters
        ----------
        X
            Term-frequency or TF-IDF pandas dataframe.

        Returns
        --------
        Input samples joint log likelihood.

        """
        unknown = [x for x in X.columns if x not in self.feature_counts]
        missing = [x for x in self.feature_counts if x not in X.columns]

        if unknown:
            X = X.drop(unknown, axis="columns")

        if missing:
            X[missing] = False

        index, columns = X.index, X.columns

        if not self.class_counts or not self.feature_counts:
            return pd.DataFrame(index=index)

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())
            X.data = X.data > self.true_threshold
        else:
            X = X > self.true_threshold

        flp = self._feature_log_prob(columns)
        neg_p = np.log(1 - np.exp(flp))

        return pd.DataFrame(
            X @ (flp - neg_p).T
            + (np.log(self.p_class_many()) + neg_p.sum(axis=1).T).values,
            index=index,
            columns=self.class_counts.keys(),
        )
