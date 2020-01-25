import copy
import functools
import itertools
import random

import pytest
from sklearn import datasets

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
            (
                linear_model.LinearRegression,
                functools.partial(
                    stream.iter_sklearn_dataset,
                    dataset=datasets.load_boston(),
                    shuffle=True,
                    random_state=42
                )
            ),
            (
                linear_model.LogisticRegression,
                functools.partial(
                    stream.iter_sklearn_dataset,
                    dataset=datasets.load_breast_cancer(),
                    shuffle=True,
                    random_state=42
                )
            )
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
                optim.initializers.Normal(mu=0, sigma=1, random_state=42)
            ]
        )
    ]
)
@pytest.mark.long
def test_finite_differences(lm, X_y):
    """Checks the gradient of a linear model via finite differences.

    References:
        1. `How to test gradient implementations <https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/>`_
        2. `Stochastic Gradient Descent Tricks <https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf>`_

    """

    scaler = preprocessing.StandardScaler()
    eps = 1e-6

    for x, y in X_y:

        x = scaler.fit_one(x).transform_one(x)

        # Store the current gradient and weights
        gradient, _ = lm._eval_gradient(x, y, 1)
        weights = lm.weights.copy()

        # d is a set of weight perturbations
        for d in iter_perturbations(weights.keys()):

            # Pertubate the weights and obtain the loss with the new weights
            lm.weights = {i: weights[i] + eps * di for i, di in d.items()}
            forward = lm.loss.eval(y_true=y, y_pred=lm._raw_dot(x))
            lm.weights = {i: weights[i] - eps * di for i, di in d.items()}
            backward = lm.loss.eval(y_true=y, y_pred=lm._raw_dot(x))

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
