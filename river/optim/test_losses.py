import importlib
import inspect
import math

import numpy as np
import pytest

from river import optim


@pytest.mark.parametrize(
    "loss",
    [
        pytest.param(loss(), id=name)
        for name, loss in inspect.getmembers(
            importlib.import_module("river.optim.losses"),
            lambda x: inspect.isclass(x)
            and not inspect.isabstract(x)
            and not issubclass(x, optim.losses.CrossEntropy),
        )
    ],
)
def test_batch_online_equivalence(loss):

    y_true = np.random.randint(2, size=30)
    y_pred = np.random.uniform(-10, 10, size=30)

    for yt, yp, g in zip(y_true, y_pred, loss.gradient(y_true, y_pred)):
        assert math.isclose(loss.gradient(yt, yp), g, abs_tol=1e-9)
