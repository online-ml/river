from __future__ import annotations

import math
import random

from river import stats


def test_weighted_variance_with_close_numbers():
    """

    Origin of this test: https://github.com/online-ml/river/issues/732

    This test would fail if Var were implemented with a numerically unstable algorithm.

    """

    D = [
        (99.99999978143265, 6),
        (99.99999989071631, 8),
        (99.99999994535816, 6),
        (99.99999997267908, 9),
        (99.99999998633952, 10),
        (99.99999999316977, 3),
        (99.99999999829245, 5),
        (99.99999999957309, 9),
    ]

    var = stats.Var()

    for x, w in D:
        var.update(x, w)

    assert var.get() > 0 and math.isclose(var.get(), 4.648047194845607e-15)


def test_revert():
    for _ in range(5):
        X = [random.random() for _ in range(20)]

        v1 = stats.Var()
        v2 = stats.Var()

        for x in X[:10]:
            v1.update(x)
            v2.update(x)

        for x in X[10:]:
            v2.update(x)
        for x in X[10:]:
            v2.revert(x)

        assert math.isclose(v1.get(), v2.get())
