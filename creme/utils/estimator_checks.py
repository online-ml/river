"""Utilities for unit testing and sanity checking estimators."""
import copy
import math
import pickle


__all__ = [
    'check_estimator',
    'guess_model'
]


def guess_model(model):

    from .. import base
    from .. import compose

    if isinstance(model, compose.Pipeline):
        return guess_model(model.final_estimator)
    elif isinstance(model, base.Wrapper):
        return guess_model(model._model)
    return model


def pick_X_y(model):

    from .. import base
    from .. import datasets

    model = guess_model(model)

    if isinstance(model, base.BinaryClassifier):
        yield from datasets.Phishing()
    elif isinstance(model, base.Regressor):
        yield from datasets.TrumpApproval()


def check_fit_one(model):

    klass = model.__class__

    for x, y in pick_X_y(model):

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        model = model.fit_one(x, y)

        # Check the model returns itself
        assert isinstance(model, klass)

        # Check fit_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one(classifier):

    for x, y in pick_X_y(classifier):

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        classifier = classifier.fit_one(x, y)
        y_pred = classifier.predict_proba_one(x)

        # Check the probabilities are coherent
        assert isinstance(y_pred, dict)
        assert math.isclose(sum(y_pred.values()), 1.)
        for proba in y_pred.values():
            assert 0. <= proba <= 1.

        # Check predict_proba_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one_binary(classifier):

    for x, y in pick_X_y(classifier):
        y_pred = classifier.predict_proba_one(x)
        classifier = classifier.fit_one(x, y)
        assert len(y_pred) == 2
        assert True in y_pred
        assert False in y_pred


def check_pickling(model):
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)


def check_repr(model):
    assert isinstance(repr(model), str)


def check_str(model):
    assert isinstance(str(model), str)


def check_debug_one(model):
    for x, y in pick_X_y(model):
        model.debug_one(x)
        model.fit_one(x, y)
        model.debug_one(x)
        break


def check_get_tags(model):
    """Checks that the ``_get_tags`` method works."""
    assert isinstance(model._get_tags(), dict)


def yield_all_checks(model):

    from .. import base

    if isinstance(model, base.AnomalyDetector):
        return
        yield

    yield check_fit_one
    yield check_pickling
    yield check_repr
    yield check_str
    yield check_get_tags
    if hasattr(model, 'debug_one'):
        yield check_debug_one

    model = guess_model(model)

    if isinstance(model, base.Classifier):
        yield check_predict_proba_one

        if not isinstance(model, base.MultiClassifier):
            yield check_predict_proba_one_binary


def check_estimator(model):
    """Check if a model adheres to ``creme``'s conventions.

    Parameters:
        model (base.Estimator)

    """

    for check in yield_all_checks(model):
        check(copy.deepcopy(model))
