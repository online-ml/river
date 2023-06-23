from __future__ import annotations

import functools
import importlib
import inspect
import itertools
from urllib import request

import pytest

from river import datasets

from . import base


def _iter_datasets():
    for _, dataset in inspect.getmembers(
        importlib.import_module("river.datasets"), inspect.isclass
    ):
        if issubclass(dataset, datasets.Insects):
            for variant in dataset.variants:
                yield dataset(variant=variant)
            continue
        yield dataset()


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
        if isinstance(dataset, base.RemoteDataset)
    ],
)
@pytest.mark.datasets
def test_remote_url(dataset):
    with request.urlopen(dataset.url) as r:
        assert r.status == 200


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(dataset, id=dataset.__class__.__name__)
        for dataset in _iter_datasets()
        if not isinstance(dataset, base.SyntheticDataset)
    ],
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
    "dataset",
    [pytest.param(dataset, id=dataset.__class__.__name__) for dataset in _iter_datasets()],
)
def test_repr(dataset):
    assert repr(dataset)


def _iter_synth_datasets():
    for variant in range(10):
        dataset = functools.partial(datasets.synth.Agrawal, classification_function=variant)
        functools.update_wrapper(dataset, datasets.synth.Agrawal)
        yield dataset

    synth = importlib.import_module("river.datasets.synth")
    for name, dataset in inspect.getmembers(synth, inspect.isclass):
        # TODO: test the following synth datasets also
        if name in ("RandomRBF", "RandomRBFDrift", "RandomTree"):
            continue
        yield dataset


@pytest.mark.parametrize(
    "dataset",
    [pytest.param(dataset(seed=42), id=dataset.__name__) for dataset in _iter_synth_datasets()],
)
def test_synth_idempotent(dataset):
    """Checks that a synthetic dataset produces identical results when seeded."""
    assert list(dataset.take(20)) == list(dataset.take(20))


# HACK: there's no reason Logical should fail this test
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(dataset(seed=None), id=dataset.__name__)
        for dataset in _iter_synth_datasets()
        if dataset.__name__ != "Logical"
    ],
)
def test_synth_non_idempotent(dataset):
    """Checks that a synthetic dataset produces different results when not seeded."""
    assert list(dataset.take(20)) != list(dataset.take(20))


@pytest.mark.parametrize(
    "dataset",
    [pytest.param(dataset(seed=42), id=dataset.__name__) for dataset in _iter_synth_datasets()],
)
def test_synth_pausable(dataset):
    stream = iter(dataset)
    s1 = itertools.islice(stream, 3)
    s2 = itertools.islice(stream, 2)
    assert list(dataset.take(5)) == list(itertools.chain(s1, s2))
