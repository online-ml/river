import copy
import math
import random

from .. import base
from .. import compose


__all__ = ['check_estimator']


def make_random_features(estimator, n_observations, n_features):
    for _ in range(n_observations):
        yield {i: random.random() for i in range(n_features)}


def make_random_targets(estimator, n_observations):

    random_funcs = {
        base.Regressor: lambda: random.random(),
        base.BinaryClassifier: lambda: random.random() > 0.5,
        base.MultiClassifier: lambda: random.choice(['a', 'b', 'c', 'd']),
        base.Transformer: lambda: None
    }

    random_func = next(func for typ, func in random_funcs.items() if isinstance(estimator, typ))

    for _ in range(n_observations):
        yield random_func()


def make_random_X_y(estimator, n_observations, n_features):
    return zip(
        make_random_features(estimator, n_observations, n_features),
        make_random_targets(estimator, n_observations)
    )


def check_fit_one(estimator):

    klass = estimator.__class__

    for x, y in make_random_X_y(estimator, n_observations=20, n_features=4):

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        estimator = estimator.fit_one(x, y)

        # Check the estimator returns itself
        assert isinstance(estimator, klass)

        # Check fit_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one(estimator):

    for x, y in make_random_X_y(estimator, n_observations=20, n_features=4):

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        estimator = estimator.fit_one(x, y)
        y_pred = estimator.predict_proba_one(x)

        # Check the probabilities are coherent
        assert isinstance(y_pred, dict)
        assert math.isclose(sum(y_pred.values()), 1.0)
        for label, proba in y_pred.items():
            assert 0.0 <= proba <= 1.0

        # Check predict_proba_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def yield_all_checks(estimator):
    yield check_fit_one

    if isinstance(estimator, (base.BinaryClassifier, base.MultiClassifier)):
        yield check_predict_proba_one


def check_estimator(estimator):

    for check in yield_all_checks(estimator):
        check(estimator)
