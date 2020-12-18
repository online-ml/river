import collections
import math

import numpy as np
import pandas as pd

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
            yes        no
    river  0.883154  0.116846
    rocks  0.952731  0.047269

    >>> model.predict_many(unseen_data)
    river    yes
    rocks    yes
    dtype: object


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
        X = (X > self.true_threshold) * 1

        agg, index = base.Groupby(keys=y).apply(np.sum, X.values)
        agg = pd.DataFrame(agg, columns=X.columns, index=index)

        self.class_counts.update(y.value_counts().to_dict())
        self.feature_counts.update((agg.T).to_dict(orient="index"))
        return self

    def p_class_many(self):
        return pd.DataFrame.from_dict(self.class_counts, orient="index").T / sum(
            self.class_counts.values()
        )

    def p_feature_given_class_many(self, columns):
        fc = collections.defaultdict(dict)
        default = {k: 0 for k, _ in self.class_counts.items()}

        for f in columns:
            for c in self.class_counts:
                count = self.feature_counts.get(f, default)
                if c in count:
                    fc[f][c] = count[c]
                else:
                    fc[f][c] = 0

        num = pd.DataFrame(fc, dtype=float).fillna(0) + self.alpha

        div = (
            pd.DataFrame.from_dict(self.class_counts, orient="index", dtype=float).T
            + self.alpha * 2
        )
        return num.div(div[num.index].T.values)

    def joint_log_likelihood_many(self, X: pd.DataFrame):
        """joint_log_likelihood optimized for mini-batch."""
        index = X.index

        unknown = []

        for x in X.columns:
            if x not in self.feature_counts:
                unknown.append(x)

        X = X.drop(unknown, axis="columns")

        X = X > self.true_threshold

        p_c = self.p_feature_given_class_many(self.feature_counts.keys())

        inverse_p_c = np.log(10e-10 + (1 - p_c))

        p_c = np.log(10e-10 + p_c)

        X[[x for x in self.feature_counts.keys() if x not in X.columns]] = False

        X = (X @ p_c.T) + ((~X) @ inverse_p_c.T).add(np.log(self.p_class_many()).values)

        X.index = index

        return X
