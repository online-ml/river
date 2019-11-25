"""Meta-models that work by wrapping other models."""
from .target_transform import BoxCoxRegressor
from .target_transform import TransformedTargetRegressor


__all__ = [
    'BoxCoxRegressor',
    'TransformedTargetRegressor'
]
