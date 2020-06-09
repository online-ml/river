import copy
import functools
import itertools
import math
import random

import numpy as np
import pandas as pd
import pytest

from creme import datasets
from creme import linear_model
from creme import optim
from creme import preprocessing
from creme import stream
from creme import utils


def iter_perturbations(keys, n=10):
    """Enumerate pertubations that will be applied to the weights."""

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
    'lm, X_y',
    [
        pytest.param(
            lm(optimizer=copy.deepcopy(optimizer), initializer=initializer, l2=0),
            X_y(),
            id=f'{lm.__name__} - {optimizer} - {initializer}'
        )
        for lm, X_y in [
            (linear_model.LinearRegression, datasets.TrumpApproval),
            (linear_model.LogisticRegression, datasets.Bananas)
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
                #optim.Momentum(),
                #optim.NesterovMomentum(),
                optim.RMSProp(),
                optim.SGD()
            ],
            [
                optim.initializers.Zeros(),
                optim.initializers.Normal(mu=0, sigma=1, seed=42)
            ]
        )
    ]
)
@pytest.mark.slow
def test_finite_differences(lm, X_y):
    """Checks the gradient of a linear model via finite differences.

    References:
        1. [How to test gradient implementations](https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)
        2. [Stochastic Gradient Descent Tricks](https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf)

    """

    scaler = preprocessing.StandardScaler()
    eps = 1e-6

    for x, y in X_y:

        print(x)
        print('---------')
        x = scaler.fit_one(x).transform_one(x)
        print(x)
        print('+++++++++')

        # Store the current gradient and weights
        gradient, _ = lm._eval_gradient_one(x, y, 1)
        weights = lm.weights.copy()

        # d is a set of weight perturbations
        for d in iter_perturbations(weights.keys()):

            # Pertubate the weights and obtain the loss with the new weights
            lm.weights = {i: weights[i] + eps * di for i, di in d.items()}
            forward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))
            lm.weights = {i: weights[i] - eps * di for i, di in d.items()}
            backward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))

            # We expect g and h to be equal
            g = utils.math.dot(d, gradient)
            h = (forward - backward) / (2 * eps)

            # Compare signs
            # TODO: reactivate this check
            #assert np.sign(g) == np.sign(h)

            # Check absolute difference
            # TODO: decrease the tolerance
            assert abs(g - h) < 1e-5

        # Reset the weights to their original values in order not to influence
        # the training loop, even though it doesn't really matter.
        lm.weights = weights
        lm.fit_one(x, y)


def test_one_many_consistent():
    """Checks that using fit_one or fit_many produces the same result."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')

    one = linear_model.LinearRegression()
    for x, y in stream.iter_pandas(X, Y):
        one.fit_one(x, y)

    many = linear_model.LinearRegression()
    for xb, yb in zip(np.array_split(X, len(X)), np.array_split(Y, len(Y))):
        many.fit_many(xb, yb)

    for i in X:
        assert math.isclose(one.weights[i], many.weights[i])


def test_shuffle_columns():
    """Checks that fit_many works identically whether columns are shuffled or not."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')

    normal = linear_model.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        normal.fit_many(xb, yb)

    shuffled = linear_model.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        cols = np.random.permutation(X.columns)
        shuffled.fit_many(xb[cols], yb)

    for i in X:
        assert math.isclose(normal.weights[i], shuffled.weights[i])


def test_add_remove_columns():
    """Checks that no exceptions are raised whenever columns are dropped and/or added."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop('five_thirty_eight')

    lin_reg = linear_model.LinearRegression()
    for xb, yb in zip(np.array_split(X, 10), np.array_split(Y, 10)):
        # Pick half of the columns at random
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        lin_reg.fit_many(xb[cols], yb)
