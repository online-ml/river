import importlib
import inspect
from urllib import request

import pytest

from creme import datasets

from . import base


def _iter_datasets():

    for variant in datasets.Insects.variants:
        yield datasets.Insects(variant=variant)

    for _, dataset in inspect.getmembers(importlib.import_module('creme.datasets'), inspect.isclass):
        if dataset.__class__.__name__ != 'Insects':
            yield dataset()


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
        if isinstance(dataset, base.RemoteDataset)
    ]
)
@pytest.mark.datasets
def test_remote_url(dataset):
    with request.urlopen(dataset.url) as r:
        assert r.status == 200


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
        if isinstance(dataset, base.RemoteDataset)
    ]
)
@pytest.mark.datasets
def test_remote_size(dataset):
    if dataset.path.is_file():
        size = dataset.path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in dataset.path.glob('**/*') if f.is_file())
    assert size == dataset.size


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
        if not isinstance(dataset, base.SyntheticDataset)
    ]
)
@pytest.mark.datasets
def test_dimensions(dataset):
    n = 0
    for x, _ in dataset:
        if not dataset.sparse:
            assert len(x) == dataset.n_features
        n += 1
    assert n == dataset.n_samples


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
    ]
)
def test_repr(dataset):
    assert repr(dataset)
