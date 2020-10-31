import copy

import pytest

from river import utils
from river import ensemble

estimator = ensemble.AdaptiveRandomForestClassifier(
    n_models=3,   # Smaller ensemble than the default to avoid bottlenecks
    seed=42
)


@pytest.mark.parametrize('estimator, check', [
    pytest.param(
        estimator,
        check,
        id=f'{estimator}:{check.__name__}'
    )
    for estimator in [
        ensemble.AdaptiveRandomForestClassifier(
            n_models=3,   # Smaller ensemble than the default to avoid bottlenecks
            seed=42
        ),
        ensemble.AdaptiveRandomForestRegressor(
            n_models=3,   # Smaller ensemble than the default to avoid bottlenecks
            seed=42
        )
    ]
    for check in utils.estimator_checks.yield_checks(estimator)
    # Skipping this test since shuffling features does impact the Adaptive Random Forest
    if check.__name__ not in {'check_shuffle_features_no_impact'}
])
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))
