"""Utilities for unit testing and sanity checking estimators."""

from __future__ import annotations

import functools
import typing

import numpy as np

from river.base import Estimator
from river.model_selection.base import ModelSelector
from river.reco.base import Ranker

from . import anomaly, clf, common, model_selection, reco, time_series

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


def _yield_datasets(model: Estimator, scale: int = 1):
    """Generates datasets for a given model.

    Parameters
    ----------
    scale
        Multiplier applied to each dataset's sample budget. The default of 1
        produces short streams that are fast enough to run on every estimator
        for every check. Pass a larger value (e.g. 10) when a check needs a
        long stream — for instance to reliably observe whether memory plateaus.

    """

    from sklearn import datasets as sk_datasets

    from river import base, compose, datasets, preprocessing, stream
    from river.anomaly.base import AnomalyDetector
    from river.time_series.base import Forecaster

    # Time series forecasters have a specialized interface: learn_one(y, x=None) and
    # forecast(horizon, xs=None). The dataset still yields (x, y) pairs so checks can
    # exercise optional exogenous features.
    if isinstance(model, Forecaster):
        yield _DummyDataset(
            ({"time": 1.0, "period": 1.0}, 10.0),
            ({"time": 2.0, "period": 2.0}, 12.0),
            ({"time": 3.0, "period": 3.0}, 13.0),
            ({"time": 4.0, "period": 0.0}, 16.0),
            ({"time": 5.0, "period": 1.0}, 18.0),
            ({"time": 6.0, "period": 2.0}, 20.0),
            ({"time": 7.0, "period": 3.0}, 21.0),
            ({"time": 8.0, "period": 0.0}, 24.0),
        )
        return

    # Recommendation models can be regressors or classifiers, but they have requirements as to the
    # structure of the data
    if isinstance(model, Ranker):
        if isinstance(model, base.Regressor):
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
    elif isinstance(model, base.MultiTargetRegressor):
        # 1
        yield stream.iter_sklearn_dataset(sk_datasets.load_linnerud())

        # 2
        class SolarFlare:
            """One-hot encoded version of `datasets.SolarFlare"""

            def __iter__(self):
                oh = (compose.SelectType(str) | preprocessing.OneHotEncoder()) + compose.SelectType(
                    int
                )
                for x, y in datasets.SolarFlare().take(200 * scale):
                    yield oh.transform_one(x), y

        yield SolarFlare()

    # Regression
    elif isinstance(model, base.Regressor):
        yield datasets.TrumpApproval().take(200 * scale)

    # Multi-output classification
    if isinstance(model, base.MultiLabelClassifier):
        yield datasets.Music().take(50 * scale)

    # Classification
    elif isinstance(model, base.Classifier):
        yield datasets.Phishing().take(200 * scale)
        yield ((x, np.bool_(y)) for x, y in datasets.Phishing().take(200 * scale))

        # Multi-class classification
        if model._multiclass and base.tags.POSITIVE_INPUT not in model._tags:  # type: ignore
            yield datasets.ImageSegments().take(200 * scale)

    # Anomaly detection
    elif isinstance(model, AnomalyDetector):
        yield datasets.CreditCard().take(1000 * scale)

    # Plain transformers (no other base class matched above). These were
    # previously uncovered by the dataset-driven checks; TrumpApproval provides
    # numeric features and a numeric target so both Transformer and
    # SupervisedTransformer can be exercised. Text-input transformers expect
    # raw strings, not dicts, and are skipped for now.
    elif isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        if base.tags.TEXT_INPUT not in model._tags:
            yield datasets.TrumpApproval().take(200 * scale)


def yield_checks(model: Estimator) -> typing.Iterator[typing.Callable]:
    """Generates unit tests for a given model.

    Parameters
    ----------
    model

    """

    from river import base, utils
    from river.anomaly.base import AnomalyDetector
    from river.time_series.base import Forecaster

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
    yield common.check_pickling_supports_roundtrip
    yield common.check_repr_roundtrips_clone
    yield common.check_clone_with_new_params_applies
    yield common.check_get_params_matches_signature
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
        common.check_radically_disappearing_features,
        common.check_predict_one_pure,
        common.check_predict_one_before_any_learn,
        common.check_no_state_aliasing_with_input,
        common.check_clone_is_independent,
    ]

    if isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        dataset_checks.append(common.check_transform_one)

    if utils.pandas.PANDAS_INSTALLED:
        if isinstance(model, (base.MiniBatchClassifier, base.MiniBatchRegressor)):
            dataset_checks.append(common.check_predict_many_matches_predict_one)
        if isinstance(model, base.MiniBatchClassifier):
            dataset_checks.append(common.check_predict_proba_many_matches_predict_proba_one)
        if isinstance(model, (base.MiniBatchTransformer, base.MiniBatchSupervisedTransformer)):
            dataset_checks.append(common.check_transform_many_matches_transform_one)

    if hasattr(model, "debug_one"):
        dataset_checks.append(common.check_debug_one)

    if model._is_stochastic:
        dataset_checks.append(common.check_seeding_is_idempotent)

    # Classifier checks
    if isinstance(model, base.Classifier) and not isinstance(model, base.MultiLabelClassifier):
        dataset_checks.append(_allow_exception(clf.check_predict_proba_one, NotImplementedError))
        dataset_checks.append(clf.check_classifier_tracks_seen_labels)
        # Specific checks for binary classifiers
        if not model._multiclass:  # type: ignore
            dataset_checks.append(
                _allow_exception(clf.check_predict_proba_one_binary, NotImplementedError)
            )

    if isinstance(model, ModelSelector):
        dataset_checks.append(model_selection.check_model_selection_order_does_not_matter)

    if isinstance(model, Ranker):
        yield reco.check_reco_routine

    if isinstance(model, AnomalyDetector):
        dataset_checks.append(anomaly.check_roc_auc)

    if isinstance(model, Forecaster):
        dataset_checks = [
            time_series.check_learn_one,
            time_series.check_forecast,
        ]

    for dataset_check in dataset_checks:
        for dataset in _yield_datasets(model):
            yield _wrapped_partial(dataset_check, dataset=dataset)

    # check_bounded_memory_growth needs a longer stream to reliably distinguish
    # accelerating growth (real) from one-time bumps (dict rehashes, Python
    # interpreter retention) — short streams let slow growers slip through.
    if not isinstance(model, Forecaster):
        for dataset in _yield_datasets(model, scale=10):
            yield _wrapped_partial(common.check_bounded_memory_growth, dataset=dataset)


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
