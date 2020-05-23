import importlib
import inspect

import pytest

from . import base


def _iter_datasets():
    return inspect.getmembers(importlib.import_module(f'creme.datasets'), inspect.isclass)


@pytest.mark.parametrize(
    'dataset',
    [
        pytest.param(dataset(), id=name)
        for name, dataset
        in _iter_datasets()
        if not isinstance(dataset, base.SyntheticDataset)
    ]
)
@pytest.mark.datasets
def test_size(dataset):
    n = 0
    for x, _ in dataset:
        if not dataset.sparse:
            assert len(x) == dataset.n_features
        n += 1
    assert n == dataset.n_samples
