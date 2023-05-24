"""Utilities for unit testing and sanity checking estimators."""
from __future__ import annotations

import functools
import typing

import numpy as np

from river.base import Estimator
from river.model_selection.base import ModelSelector
from river.reco.base import Ranker

from . import anomaly, clf, common, model_selection, reco

__all__ = ["check_estimator", "yield_checks"]


def _allow_exception(func, exception):
    def f(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except exception:
            pass

    f.__name__ = func.__name__
    return f


def _wrapped_partial(func, *args, **kwargs):
    """

    Taken from http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/

    """
    partial = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial, func)
    return partial


class _DummyDataset:
    def __init__(self, *data):
        self.data = data

    def __iter__(self):
        yield from self.data


def _yield_datasets(model: Estimator):
    """Generates datasets for a given model."""

    from sklearn import datasets as sk_datasets

    from river import base, compose, datasets, preprocessing, stream, utils

    # Recommendation models can be regressors or classifiers, but they have requirements as to the
    # structure of the data
    if isinstance(utils.inspect.extract_relevant(model), Ranker):
        if utils.inspect.isregressor(model):
            yield _DummyDataset(
                ("Alice", "Superman", 8),
                ("Alice", "Terminator", 9),
                ("Alice", "Star Wars", 8),
                ("Alice", "Notting Hill", 2),
                ("Alice", "Harry Potter", 5),
                ("Bob", "Superman", 8),
                ("Bob", "Terminator", 9),
                ("Bob", "Star Wars", 8),
                ("Bob", "Notting Hill", 2),
            )
        return

    # Multi-output regression
    elif utils.inspect.ismoregressor(model):
        # 1
        yield stream.iter_sklearn_dataset(sk_datasets.load_linnerud())

        # 2
        class SolarFlare:
            """One-hot encoded version of `datasets.SolarFlare"""

            def __iter__(self):
                oh = (compose.SelectType(str) | preprocessing.OneHotEncoder()) + compose.SelectType(
                    int
                )
                for x, y in datasets.SolarFlare().take(200):
                    yield oh.transform_one(x), y

        yield SolarFlare()

    # Regression
    elif utils.inspect.isregressor(model):
        yield datasets.TrumpApproval().take(200)

    # Multi-output classification
    if utils.inspect.ismoclassifier(model):
        yield datasets.Music().take(50)

    # Classification
    elif utils.inspect.isclassifier(model):
        yield datasets.Phishing().take(200)
        yield ((x, np.bool_(y)) for x, y in datasets.Phishing().take(200))

        # Multi-class classification
        if model._multiclass and base.tags.POSITIVE_INPUT not in model._tags:  # type: ignore
            yield datasets.ImageSegments().take(200)

    # Anomaly detection
    elif utils.inspect.isanomalydetector(model):
        yield datasets.CreditCard().take(1000)


def yield_checks(model: Estimator) -> typing.Iterator[typing.Callable]:
    """Generates unit tests for a given model.

    Parameters
    ----------
    model

    """

    from river import base, utils

    # General checks
    yield common.check_repr
    yield common.check_str
    yield common.check_tags
    yield common.check_clone_same_class
    yield common.check_clone_is_idempotent
    yield common.check_init_has_default_params_for_tests
    yield common.check_init_default_params_are_not_mutable
    yield common.check_doc
    yield common.check_clone_changes_memory_addresses
    yield common.check_mutate_can_be_idempotent
    if model._mutable_attributes:
        yield common.check_mutable_attributes_exist

    if isinstance(model, base.Wrapper):
        yield common.check_wrapper_accepts_kwargs

    # Checks that make use of datasets
    dataset_checks = [
        common.check_learn_one,
        common.check_pickling,
        common.check_shuffle_features_no_impact,
        common.check_emerging_features,
        common.check_disappearing_features,
    ]

    if hasattr(model, "debug_one"):
        dataset_checks.append(common.check_debug_one)

    if model._is_stochastic:
        dataset_checks.append(common.check_seeding_is_idempotent)

    # Classifier checks
    if utils.inspect.isclassifier(model) and not utils.inspect.ismoclassifier(model):
        dataset_checks.append(_allow_exception(clf.check_predict_proba_one, NotImplementedError))
        # Specific checks for binary classifiers
        if not model._multiclass:  # type: ignore
            dataset_checks.append(
                _allow_exception(clf.check_predict_proba_one_binary, NotImplementedError)
            )

    if isinstance(utils.inspect.extract_relevant(model), ModelSelector):
        dataset_checks.append(model_selection.check_model_selection_order_does_not_matter)

    if isinstance(utils.inspect.extract_relevant(model), Ranker):
        yield reco.check_reco_routine

    if utils.inspect.isanomalydetector(model):
        dataset_checks.append(anomaly.check_roc_auc)

    for dataset_check in dataset_checks:
        for dataset in _yield_datasets(model):
            yield _wrapped_partial(dataset_check, dataset=dataset)


def check_estimator(model: Estimator):
    """Check if a model adheres to River's conventions.

    This will run a series of unit tests. The nature of the unit tests depends on the type of
    model.

    Parameters
    ----------
    model

    """
    for check in yield_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(model.clone())
