from __future__ import annotations

import importlib
import inspect
import math
import random
import typing

import numpy as np
import pytest

from river import optim, utils


def losses() -> typing.Iterable[optim.losses.Loss]:
    for _, loss in inspect.getmembers(
        importlib.import_module("river.optim.losses"),
        lambda x: (
            inspect.isclass(x)
            and not inspect.isabstract(x)
            and not issubclass(x, optim.losses.CrossEntropy)
        ),
    ):
        yield loss()


@pytest.mark.parametrize(
    "loss",
    [pytest.param(loss, id=loss.__class__.__name__) for loss in losses()],
)
def test_loss_batch_online_equivalence(loss):
    y_true = np.random.randint(2, size=30)
    y_pred = np.random.uniform(-10, 10, size=30)

    for yt, yp, g in zip(y_true, y_pred, loss.gradient(y_true, y_pred)):
        assert math.isclose(loss.gradient(yt, yp), g, abs_tol=1e-9)


def optimizers() -> typing.Iterable[optim.base.Optimizer]:
    for _, optimizer in inspect.getmembers(
        importlib.import_module("river.optim"),
        lambda x: (
            inspect.isclass(x)
            and issubclass(x, optim.base.Optimizer)
            and x is not optim.base.Optimizer
        ),
    ):
        for params in optimizer._unit_test_params():
            yield optimizer(**params)


@pytest.mark.parametrize(
    "optimizer",
    [pytest.param(optimizer, id=optimizer.__class__.__name__) for optimizer in optimizers()],
)
def test_optimizer_step_with_dict_same_as_step_with_vector_dict(optimizer):
    w_dict = {i: random.uniform(-5, 5) for i in range(10)}
    w_vector = utils.VectorDict(w_dict)

    g_dict = {i: random.uniform(-5, 5) for i in range(10)}
    g_vector = utils.VectorDict(g_dict)

    w_dict = optimizer._step_with_dict(w_dict, g_dict)
    try:
        w_vector = optimizer.clone()._step_with_vector(w_vector, g_vector)
    except NotImplementedError:
        pytest.skip("step_with_vector not implemented")

    for i, w in w_vector.to_dict().items():
        assert math.isclose(w, w_dict[i])


@pytest.mark.parametrize(
    "optimizer",
    [pytest.param(optimizer, id=optimizer.__class__.__name__) for optimizer in optimizers()],
)
def test_optimizer_step_with_dict_same_as_step_with_numpy_array(optimizer):
    w_dict = {i: random.uniform(-5, 5) for i in range(10)}
    w_vector = np.asarray(list(w_dict.values()))

    g_dict = {i: random.uniform(-5, 5) for i in range(10)}
    g_vector = np.asarray(list(g_dict.values()))

    w_dict = optimizer._step_with_dict(w_dict, g_dict)
    try:
        w_vector = optimizer.clone()._step_with_vector(w_vector, g_vector)
    except NotImplementedError:
        pytest.skip("step_with_vector not implemented")

    for i, w in dict(enumerate(w_vector)).items():
        assert math.isclose(w, w_dict[i])


def test_newton_maintains_true_inverse_hessian():
    """Newton must keep `_H_inv` equal to inv(eps * I + sum_t g_t g_t^T).

    This pins both the Sherman-Morrison update and, crucially, the inverse-Hessian
    initialization: starting from `eps * I` (instead of `(1 / eps) * I`) would break the
    invariant immediately.
    """
    rng = random.Random(42)
    eps = 1e-3
    optimizer = optim.Newton(eps=eps)
    d = 5

    hessian = np.eye(d) * eps
    for _ in range(50):
        g = {i: rng.uniform(-3, 3) for i in range(d)}
        optimizer._step_with_dict({i: 0.0 for i in range(d)}, g)
        g_arr = np.array([g[i] for i in range(d)])
        hessian += np.outer(g_arr, g_arr)

    # `_H_inv` is allocated with spare capacity; the active block holds the inverse Hessian
    # while untouched features keep the (1 / eps) prior on the diagonal.
    assert np.allclose(optimizer._H_inv[:d, :d], np.linalg.inv(hessian))
