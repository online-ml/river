"""
Meta-estimators that wrap other estimators.
"""
from .detrender import Detrender
from .detrender import GroupDetrender


__all__ = ['Detrender', 'GroupDetrender']
