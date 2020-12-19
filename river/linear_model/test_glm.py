import copy
import itertools
import math
import random

import numpy as np
import pandas as pd
from sklearn import linear_model as sklm
import pytest

from river import datasets
from river import linear_model as lm
from river import optim
from river import preprocessing
from river import stream
from river import utils


def iter_perturbations(keys, n=10):
    """Enumerate perturbations that will be applied to the weights."""

    # Enumerate unit vectors
    for i in keys:
        yield {j: int(i == j) for j in keys}

    # Enumerate samples from a spherical Gaussian
    for _ in range(n):
        p = {j: random.gauss(0, 1) for j in keys}
        norm = utils.math.norm(p, order=2)
        for j in p:
            p[j] /= norm
        yield p


@pytest.mark.parametrize(
    "lm, dataset",
    [
        pytest.param(
            lm(optimizer=copy.deepcopy(optimizer), initializer=initializer, l2=0),
            dataset,
            id=f"{lm.__name__} - {optimizer} - {initializer}",
        )
        for lm, dataset in [
            (lm.LinearRegression, datasets.TrumpApproval()),
            (lm.LogisticRegression, datasets.Bananas()),
        ]
        for optimizer, initializer in itertools.product(
            [
                optim.AdaBound(),
                optim.AdaDelta(),
                optim.AdaGrad(),
                optim.AdaMax(),
                optim.Adam(),
                optim.AMSGrad(),
                # TODO: check momentum optimizers
                # optim.Momentum(),
                # optim.NesterovMomentum(),
                optim.RMSProp(),
                optim.SGD(),
            ],
            [
                optim.initializers.Zeros(),
                optim.initializers.Normal(mu=0, sigma=1, seed=42),
            ],
        )
    ],
)
@pytest.mark.slow
def test_finite_differences(lm, dataset):
    """Checks the gradient of a linear model via finite differences.

    References
    ----------
    [^1]: [How to test gradient implementations](https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)
    [^2]: [Stochastic Gradient Descent Tricks](https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf)

    """

    scaler = preprocessing.StandardScaler()
    eps = 1e-6

    for x, y in dataset:

        x = scaler.learn_one(x).transform_one(x)

        # Store the current gradient and weights
        gradient, _ = lm._eval_gradient_one(x, y, 1)
        weights = copy.deepcopy(lm._weights)

        # d is a set of weight perturbations
        for d in iter_perturbations(weights.keys()):

            # Pertubate the weights and obtain the loss with the new weights
            lm._weights = utils.VectorDict({i: weights[i] + eps * di for i, di in d.items()})
            forward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))
            lm._weights = utils.VectorDict({i: weights[i] - eps * di for i, di in d.items()})
            backward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))

            # We expect g and h to be equal
            g = utils.math.dot(d, gradient)
            h = (forward - backward) / (2 * eps)

            # Compare signs
            # TODO: reactivate this check
            # assert np.sign(g) == np.sign(h)

            # Check absolute difference
            # TODO: decrease the tolerance
            assert abs(g - h) < 1e-5

        # Reset the weights to their original values in order not to influence
        # the training loop, even though it doesn't really matter.
        lm._weights = weights
        lm.learn_one(x, y)


def test_one_many_consistent():
    """Checks that using learn_one or learn_many produces the same result."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    one = lm.LinearRegression()
    for x, y in stream.iter_pandas(X, Y):
        one.learn_one(x, y)

    many = lm.LinearRegression()
    for xb, yb in zip(np.array_split(X, len(X)), np.array_split(Y, len(Y))):
        many.learn_many(xb, yb)

    for i in X:
        assert math.isclose(one.weights[i], many.weights[i])


def test_shuffle_columns():
    """Checks that learn_many works identically whether columns are shuffled or not."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    normal = lm.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        normal.learn_many(xb, yb)

    shuffled = lm.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        cols = np.random.permutation(X.columns)
        shuffled.learn_many(xb[cols], yb)

    for i in X:
        assert math.isclose(normal.weights[i], shuffled.weights[i])


def test_add_remove_columns():
    """Checks that no exceptions are raised whenever columns are dropped and/or added."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    lin_reg = lm.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        # Pick half of the columns at random
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        lin_reg.learn_many(xb[cols], yb)


def test_lin_reg_sklearn_coherence():
    """Checks that the sklearn and river implementations produce the same results."""

    class SquaredLoss:
        """sklearn removes the leading 2 from the gradient of the squared loss."""

        def gradient(self, y_true, y_pred):
            return y_pred - y_true

    ss = preprocessing.StandardScaler()
    cr = lm.LinearRegression(optimizer=optim.SGD(0.01), loss=SquaredLoss())
    sk = sklm.SGDRegressor(learning_rate="constant", eta0=0.01, alpha=0.0)

    for x, y in datasets.TrumpApproval():
        x = ss.learn_one(x).transform_one(x)
        cr.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y])

    for i, w in enumerate(cr.weights.values()):
        assert math.isclose(w, sk.coef_[i])

    assert math.isclose(cr.intercept, sk.intercept_[0])


def test_log_reg_sklearn_coherence():
    """Checks that the sklearn and river implementations produce the same results."""

    ss = preprocessing.StandardScaler()
    cr = lm.LogisticRegression(optimizer=optim.SGD(0.01))
    sk = sklm.SGDClassifier(learning_rate="constant", eta0=0.01, alpha=0.0, loss="log")

    for x, y in datasets.Bananas():
        x = ss.learn_one(x).transform_one(x)
        cr.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])

    for i, w in enumerate(cr.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])

    assert math.isclose(cr.intercept, sk.intercept_[0])


def test_perceptron_sklearn_coherence():
    """Checks that the sklearn and river implementations produce the same results."""

    ss = preprocessing.StandardScaler()
    cr = lm.Perceptron()
    sk = sklm.Perceptron()

    for x, y in datasets.Bananas():
        x = ss.learn_one(x).transform_one(x)
        cr.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])

    for i, w in enumerate(cr.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])

    assert math.isclose(cr.intercept, sk.intercept_[0])
