import importlib
import inspect
import urllib.request

import pytest


def _iter_datasets():
    return inspect.getmembers(importlib.import_module(f'creme.datasets'), inspect.isclass)


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset(), id=name)
        for name, dataset
        in _iter_datasets()
        if dataset()._remote
    ]
)
@pytest.mark.web
def test_remote_url_200(dataset):
    r = urllib.request.urlopen(dataset.dl_params['url'])
    assert r.getcode() == 200


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset(), id=name)
        for name, dataset
        in _iter_datasets()
    ]
)
@pytest.mark.web
def test_size(dataset):
    n = 0
    for x, _ in dataset:
        assert len(x) == dataset.n_features
        n += 1
    assert n == dataset.n_samples
