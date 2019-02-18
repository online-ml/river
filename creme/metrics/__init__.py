"""
A set of metrics used in machine learning that can be computed in a streaming
fashion.
"""
from .mse import MSE
from .rmse import RMSE
from .rmsle import RMSLE


__all__ = [
    'MSE',
    'RMSE',
    'RMSLE'
]
