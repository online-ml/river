import collections
import math

import pandas as pd
import numpy as np

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
            yes        no
    river  0.553531  0.446469
    rocks  0.881499  0.118501

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

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        agg, index = base.Groupby(keys=y).apply(np.sum, X.values)
        agg = pd.DataFrame(agg, columns=X.columns, index=index)

        self.feature_counts.update((agg.T).to_dict(orient="index"))
        self.class_counts.update(y.value_counts().to_dict())
        self.class_totals.update(agg.sum(axis="columns").to_dict())

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

        for f in columns:
            fc[f] = self.feature_counts.get(f, default)

        num = pd.DataFrame(fc, dtype=float).fillna(0) + self.alpha
        div = (
            pd.DataFrame.from_dict(self.class_totals, orient="index", dtype=float).T
            + self.alpha * self.n_terms
        )
        return num.div(div[num.index].T.values)

    def joint_log_likelihood_many(self, X: pd.DataFrame):
        pfc = self.p_feature_given_class_many(X.columns)
        p_class = np.log(self.p_class_many())[pfc.index]
        p = X @ np.log(pfc).values.T
        p.columns = pfc.index
        return p.add(p_class.values, axis="columns")
