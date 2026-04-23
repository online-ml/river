"""Neural networks."""

from __future__ import annotations

import warnings

from . import activations
from .mlp import MLPRegressor

warnings.warn(
    "`river.neural_net` is deprecated and will be removed in a future release; "
    "use `sklearn` for neural networks instead.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["activations", "MLPRegressor"]
