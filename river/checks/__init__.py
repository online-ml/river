"""Utilities for unit testing and sanity checking estimators."""

import copy
import functools
import inspect
import itertools
import math
import pickle
import random

import numpy as np

from river.base import Estimator
from river.model_selection import ModelSelector
from river.reco import Recommender

from .base import *
from .clf import *
from .model_selection import *
from .reco import *


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


def _yield_datasets(model: Estimator):
    """Generates datasets for a given model."""

    from sklearn import datasets as sk_datasets

    from river import base, compose, datasets, preprocessing, stream, utils

    # Multi-output regression
    if utils.inspect.ismoregressor(model):

        # 1
        yield stream.iter_sklearn_dataset(sk_datasets.load_linnerud())

        # 2
        class SolarFlare:
            """One-hot encoded version of `datasets.SolarFlare"""

            def __iter__(self):
                oh = (
                    compose.SelectType(str) | preprocessing.OneHotEncoder()
                ) + compose.SelectType(int)
                for x, y in datasets.SolarFlare().take(200):
                    yield oh.transform_one(x), y

        yield SolarFlare()

    # Regression
    elif utils.inspect.isregressor(model):
        yield datasets.TrumpApproval().take(200)

    # Multi-output classification
    if utils.inspect.ismoclassifier(model):
        yield datasets.Music().take(200)

    # Classification
    elif utils.inspect.isclassifier(model):

        yield datasets.Phishing().take(200)
        yield ((x, np.bool_(y)) for x, y in datasets.Phishing().take(200))

        # Multi-class classification
        if model._multiclass and base.tags.POSITIVE_INPUT not in model._tags:
            yield datasets.ImageSegments().take(200)


def yield_checks(model: Estimator):
    """Generates unit tests for a given model."""

    from river import model_selection, utils

    # General checks
    yield check_repr
    yield check_str
    yield check_tags
    yield check_set_params_idempotent
    yield check_init_has_default_params_for_tests
    yield check_init_default_params_are_not_mutable
    yield check_doc
    yield check_clone

    if utils.inspect.isclassifier(model):
        yield check_multiclass_is_bool

    # Checks that make use of datasets
    checks = [
        check_learn_one,
        check_pickling,
        check_shuffle_features_no_impact,
        check_emerging_features,
        check_disappearing_features,
    ]

    if hasattr(model, "debug_one"):
        checks.append(check_debug_one)

    if model._is_stochastic:
        checks.append(check_seeding_is_idempotent)

    # Classifier checks
    if utils.inspect.isclassifier(model) and not utils.inspect.ismoclassifier(model):
        checks.append(_allow_exception(check_predict_proba_one, NotImplementedError))
        # Specific checks for binary classifiers
        if not model._multiclass:
            checks.append(
                _allow_exception(check_predict_proba_one_binary, NotImplementedError)
            )

    if isinstance(utils.inspect.extract_relevant(model), ModelSelector):
        checks.append(check_model_selection_order_does_not_matter)

    if isinstance(utils.inspect.extract_relevant(model), Recommender):
        yield check_reco_routine

    for check in checks:
        for dataset in _yield_datasets(model):
            yield _wrapped_partial(check, dataset=dataset)


def check_estimator(model: Estimator):
    """Check if a model adheres to `river`'s conventions.

    This will run a series of unit tests. The nature of the unit tests depends on the type of
    model.

    Parameters
    ----------
    model

    """
    for check in yield_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(copy.deepcopy(model))
