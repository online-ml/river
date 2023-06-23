from __future__ import annotations

import inspect
import math


def assert_predictions_are_close(y1, y2):
    if isinstance(y1, dict):
        for k in y1:
            assert_predictions_are_close(y1[k], y2[k])
    elif isinstance(y1, float):
        assert math.isclose(y1, y2, rel_tol=1e-06)
    else:
        assert y1 == y2


def seed_params(params, seed):
    """Looks for "seed" keys and sets the value."""

    def is_class_param(param):
        return isinstance(param, tuple) and inspect.isclass(param[0]) and isinstance(param[1], dict)

    if is_class_param(params):
        return params[0], seed_params(params[1], seed)

    if not isinstance(params, dict):
        return params

    return {
        name: seed if name == "seed" else seed_params(param, seed) for name, param in params.items()
    }
