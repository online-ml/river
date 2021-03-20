import collections
import copy

from river import base, linear_model
from river.utils.math import prod
from river.utils.skmultiflow_utils import check_random_state

__all__ = [
    "ClassifierChain",
    "RegressorChain",
    "ProbabilisticClassifierChain",
    "MonteCarloClassifierChain",
]


class BaseChain(base.WrapperMixin, collections.UserDict):
    def __init__(self, model, order: list = None):
        super().__init__()
        self.model = model
        self.order = order or []

    @property
    def _wrapped_model(self):
        return self.model

    def __getitem__(self, key):
        try:
            return collections.UserDict.__getitem__(self, key)
        except KeyError:
            collections.UserDict.__setitem__(self, key, copy.deepcopy(self.model))
            return self[key]


class ClassifierChain(BaseChain, base.Classifier, base.MultiOutputMixin):
    """A multi-output model that arranges classifiers into a chain.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second model. The prediction for the second output will be used as a feature
    for the third model, etc. This "chain model" is therefore capable of capturing dependencies
    between outputs.

    Parameters
    ----------
    model
    order
        A list with the targets order in which to construct the chain. If `None` then the order
        will be inferred from the order of the keys in the target.


    Examples
    --------

    >>> from river import feature_selection
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.fetch_openml('yeast', version=4, as_frame=False),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = feature_selection.VarianceThreshold(threshold=0.01)
    >>> model |= preprocessing.StandardScaler()
    >>> model |= multioutput.ClassifierChain(
    ...     model=linear_model.LogisticRegression(),
    ...     order=list(range(14))
    ... )

    >>> metric = metrics.Jaccard()

    >>> for x, y in dataset:
    ...     # Convert y values to booleans
    ...     y = {i: yi == 'TRUE' for i, yi in y.items()}
    ...     y_pred = model.predict_one(x)
    ...     metric = metric.update(y, y_pred)
    ...     model = model.learn_one(x, y)

    >>> metric
    Jaccard: 0.451524

    References
    ----------
    [^1]: [Multi-Output Chain Models and their Application in Data Streams](https://jmread.github.io/talks/2019_03_08-Imperial_Stats_Seminar.pdf)

    """

    def __init__(self, model: base.Classifier, order: list = None):
        super().__init__(model, order)

    @classmethod
    def _unit_test_params(cls):
        return {"model": linear_model.LogisticRegression()}

    def _multiclass(self):
        return self.model._multiclass

    def learn_one(self, x, y):

        x = copy.copy(x)
        n_seen = 0

        for o in self.order:
            clf = self[o]

            # Make predictions before the model is updated to avoid leakage
            y_pred = clf.predict_proba_one(x)

            # We handle the case where an output has been seen in the past but is missing now
            try:
                y_o = y[o]
                n_seen += 1
                clf.learn_one(x, y_o)
            except KeyError:
                pass

            # The predictions are stored as features for the next label
            if clf._multiclass:
                for label, proba in y_pred.items():
                    x[f"{o}_{label}"] = proba
            else:
                x[o] = y_pred[True]

        # Now we check if there are any new outputs
        n_unseen = len(y) - n_seen
        if n_unseen:
            for o in y:
                if o not in self.order:
                    self.order.append(o)

        return self

    def predict_proba_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        for o in self.order:
            clf = self[o]

            y_pred[o] = clf.predict_proba_one(x)

            # The predictions are stored as features for the next label
            if clf._multiclass:
                for label, proba in y_pred.items():
                    x[f"{o}_{label}"] = proba
            else:
                x[o] = y_pred[o][True]

        return y_pred

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        return {c: max(y_pred[c], key=y_pred[c].get) for c in y_pred}


class RegressorChain(BaseChain, base.Regressor, base.MultiOutputMixin):
    """A multi-output model that arranges regressor into a chain.

    This will create one model per output. The prediction of the first output will be used as a
    feature in the second output. The prediction for the second output will be used as a feature
    for the third, etc. This "chain model" is therefore capable of capturing dependencies between
    outputs.

    Parameters
    ----------
    model
    order
        A list with the targets order in which to construct the chain. If `None` then the order
        will be inferred from the order of the keys in the target.

    Examples
    --------

    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_linnerud(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = multioutput.RegressorChain(
    ...     model=(
    ...         preprocessing.StandardScaler() |
    ...         linear_model.LinearRegression(intercept_lr=0.3)
    ...     ),
    ...     order=[0, 1, 2]
    ... )

    >>> metric = metrics.RegressionMultiOutput(metrics.MAE())

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 12.649592

    """

    def __init__(self, model: base.Regressor, order: list = None):
        super().__init__(model, order)

    @classmethod
    def _unit_test_params(cls):
        return {"model": linear_model.LinearRegression()}

    def learn_one(self, x, y):

        x = copy.copy(x)
        n_seen = 0

        for o in self.order:
            reg = self[o]

            # Make predictions before the model is updated to avoid leakage
            y_pred = reg.predict_one(x)

            # We handle the case where an output has been seen in the past but is missing now
            try:
                y_o = y[o]
                n_seen += 1
                reg.learn_one(x, y_o)
            except KeyError:
                pass

            # The predictions are stored as features for the next label
            x[o] = y_pred

        # Now we check if there are any new outputs
        n_unseen = len(y) - n_seen
        if n_unseen:
            for o in y:
                if o not in self.order:
                    self.order.append(o)

        return self

    def predict_one(self, x):

        x = copy.copy(x)
        y_pred = {}

        if not isinstance(self.order, list):
            return y_pred

        for o, clf in self.items():
            y_pred[o] = clf.predict_one(x)
            x[o] = y_pred[o]

        return y_pred


class ProbabilisticClassifierChain(ClassifierChain):
    r"""Probabilistic Classifier Chains.

    The Probabilistic Classifier Chains (PCC) [^1] is a Bayes-optimal method
    based on the Classifier Chains (CC).

    Consider the concept of chaining classifiers as searching a path in a
    binary tree whose leaf nodes are associated with a label $y \in Y$. While
    CC searches only a single path in the aforementioned binary tree, PCC looks
    at each of the $2^l$ paths, where $l$ is the number of labels. This limits
    the applicability of the method to data sets with a small to moderate
    number of labels. The authors recommend no more than about 15 labels for
    real-world applications.

    Parameters
    ----------
    model

    Examples
    --------
    >>> from river import feature_selection
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import synth

    >>> dataset = synth.Logical(seed=42, n_tiles=100)

    >>> model = multioutput.ProbabilisticClassifierChain(
    ...     model=linear_model.LogisticRegression()
    ... )

    >>> metric = metrics.Jaccard()

    >>> for x, y in dataset:
    ...    y_pred = model.predict_one(x)
    ...    metric = metric.update(y, y_pred)
    ...    model = model.learn_one(x, y)

    >>> metric
    Jaccard: 0.571429

    References
    ----------

    [^1]: Cheng, W., Hüllermeier, E., & Dembczynski, K. J. (2010).
          Bayes optimal multilabel classification via probabilistic classifier
          chains. In Proceedings of the 27th international conference on
          machine learning (ICML-10) (pp. 279-286).

    """

    def __init__(self, model: base.Classifier):
        super().__init__(model)

    def predict_one(self, x):
        y_pred = {}

        if not isinstance(self.order, list):
            return y_pred

        max_payoff = 0.0
        n_labels = len(self.order)
        # for each and every possible label combination
        for label in range(2 ** n_labels):
            # put together a binary label vector
            y_gen = {
                i: int(v)
                for i, v in zip(self.order, list(bin(label)[2:].zfill(n_labels)))
            }
            # ... and gauge a probability for it (given x)
            payoff = self._payoff(x=x, y=y_gen)
            # if it performs well, keep it, and record the max
            if payoff > max_payoff:
                y_pred = copy.copy(y_gen)
                max_payoff = payoff
        return y_pred

    def _payoff(self, x, y):
        # Calculate payoff for predicting y | x, under the chains model.
        p = {}

        x = copy.copy(x)

        for label in self.order:
            clf = self[label]

            y_pred = clf.predict_proba_one(x)
            # Extend features
            x[label] = y[label]
            p[label] = y_pred[y[label]]

        return prod(p.values())


class MonteCarloClassifierChain(ProbabilisticClassifierChain):
    """Monte Carlo Sampling Classifier Chains.

    Probabilistic Classifier Chains using Monte Carlo sampling, as
    described in [^1].

    m samples are taken from the posterior distribution. Therefore we
    need a probabilistic interpretation of the output, and thus, this is a
    particular variety of ProbabilisticClassifierChain.

    Parameters
    ----------
    model
    m
        Number of samples to take from the posterior distribution.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> from river import feature_selection
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import multioutput
    >>> from river import preprocessing
    >>> from river import synth

    >>> dataset = synth.Logical(seed=42, n_tiles=100)

    >>> model = multioutput.MonteCarloClassifierChain(
    ...     model=linear_model.LogisticRegression(),
    ...     m=10,
    ...     seed=42
    ... )

    >>> metric = metrics.Jaccard()

    >>> for x, y in dataset:
    ...    y_pred = model.predict_one(x)
    ...    metric = metric.update(y, y_pred)
    ...    model = model.learn_one(x, y)

    >>> metric
    Jaccard: 0.568087

    References
    ----------
    [^1]: Read, J., Martino, L., & Luengo, D. (2014). Efficient monte carlo
          methods for multi-dimensional learning with classifier chains.
          Pattern Recognition, 47(3), 1535-1546.

    """

    def __init__(self, model: base.Classifier, m: int = 10, seed: int = None):
        ClassifierChain.__init__(self, model=model, order=None)
        self.seed = seed
        self._rng = check_random_state(seed)
        self.m = m

    def _sample(self, x):
        # Sample y ~ P(y|x)
        p = {}
        y = {}
        x = copy.copy(x)

        for label in self.order:
            clf = self[label]

            y_pred = clf.predict_proba_one(x)
            y_val = self._rng.choice(2, 1, p=[v for v in y_pred.values()])[0]
            # Extend features
            x[label] = y_val
            y[label] = y_val
            p[label] = y_pred[y_val]

        return y, p

    def predict_one(self, x):
        y_pred = {}

        if not isinstance(self.order, list):
            return y_pred

        y_pred = ClassifierChain.predict_one(self, x)
        max_payoff = self._payoff(x=x, y=y_pred)
        # for M times
        for m in range(self.m):
            y_, p_ = self._sample(
                x
            )  # N.B. in fact, the calculation p_ is done again in P.
            payoff = self._payoff(x=x, y=y_)
            # if it performs well, keep it, and record the max
            if payoff > max_payoff:
                y_pred = copy.copy(y_)
                max_payoff = payoff
        return y_pred
