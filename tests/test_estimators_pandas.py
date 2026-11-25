"""Dedicated tests that need Pandas, for the mini-batch estimators."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from sklearn import linear_model as sk_linear_model

from river import (
    anomaly,
    base,
    compat,
    feature_extraction,
    linear_model,
    multiclass,
    preprocessing,
)
from river.checks import _wrapped_partial, _yield_datasets, common

from .test_estimators import iter_estimators_which_can_be_tested


def _yield_pandas_checks(model) -> Iterator[Callable]:
    dataset_checks = []
    if isinstance(model, (base.MiniBatchClassifier, base.MiniBatchRegressor)):
        dataset_checks.append(common.check_predict_many_matches_predict_one)
        dataset_checks.append(common.check_learn_many_matches_learn_one)
    if isinstance(model, base.MiniBatchClassifier):
        dataset_checks.append(common.check_predict_proba_many_matches_predict_proba_one)
    if isinstance(model, (base.MiniBatchTransformer, base.MiniBatchSupervisedTransformer)):
        dataset_checks.append(common.check_transform_many_matches_transform_one)
        dataset_checks.append(common.check_learn_many_matches_learn_one)

    for dataset_check in dataset_checks:
        for dataset in _yield_datasets(model):
            yield _wrapped_partial(dataset_check, dataset=dataset)


# For testing efficiency reasons, River splits its tests into two categories: those that need the Pandas library to run, and those that do not. This detection of Pandas use is done by looking up if an indication of the use of Pandas exists in the file; see `_PANDAS_NEEDLES` in `conftest.py`.
# Crucially, this Pandas detection does not propagate for indirect uses. For a file to be detected as requiring Pandas, it must include one of the detected strings.
# This file aims to run all the checks that require Pandas (mostly mini-batch tests). Because of that, it needs to be detected as using Pandas, even if it does not import pandas directly.
# Hence, we need to write in this file a string that's detected as a dependency on Pandas. We'll to it here: "import pandas".
@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(
            estimator,
            check,
            id=f"{estimator}:{check.__name__}",
        )
        for estimator in list(iter_estimators_which_can_be_tested())
        + [
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            preprocessing.StandardScaler() | linear_model.PAClassifier(),
            (
                preprocessing.StandardScaler()
                | preprocessing.TargetStandardScaler(
                    regressor=linear_model.LinearRegression(),
                )
            ),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
            ),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.PAClassifier())
            ),
            preprocessing.MinMaxScaler() + preprocessing.StandardScaler(),
            (
                feature_extraction.PolynomialExtender()
                | preprocessing.StandardScaler()
                | linear_model.LinearRegression()
            ),
            preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(),
            (
                preprocessing.StandardScaler()
                | compat.convert_sklearn_to_river(sk_linear_model.SGDRegressor(tol=1e-10))
            ),
        ]
        for check in _yield_pandas_checks(estimator)
        if check.__name__ not in estimator._unit_test_skips()
    ],
)
def test_check_estimator(estimator, check):
    check(estimator.clone())
