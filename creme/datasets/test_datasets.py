import importlib
import inspect

import pytest


@pytest.mark.parametrize(
    'func',
    [
        pytest.param(func, id=name)
        for name, func
        in inspect.getmembers(importlib.import_module(f'creme.datasets'), inspect.isfunction)
        if name.startswith('fetch')
    ]
)
@pytest.mark.web
def test_fetch(func):
    X_y = func()
    x, y = next(X_y)


@pytest.mark.parametrize(
    'func',
    [
        pytest.param(func, id=name)
        for name, func
        in inspect.getmembers(importlib.import_module(f'creme.datasets'), inspect.isfunction)
        if name.startswith('load')
    ]
)
def test_load(func):
    X_y = func()
    x, y = next(X_y)
