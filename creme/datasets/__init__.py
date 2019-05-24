from .airline import load_airline
from .bikes import fetch_bikes
from .electricity import fetch_electricity
from .restaurants import fetch_restaurants


__all__ = [
    'fetch_bikes',
    'fetch_electricity',
    'fetch_restaurants',
    'load_airline'
]
