"""Utilities for unit testing and sanity checking estimators."""
import copy
import functools
import math
import pickle
import random

import numpy as np


__all__ = ["check_estimator"]


def yield_datasets(model):

    from river import base
    from river import compose
    from river import datasets
    from river import preprocessing
    from river import stream
    from river import utils
    from sklearn import datasets as sk_datasets

    # Multi-output regression
    if utils.inspect.ismoregressor(model):

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
        yield datasets.Music().take(200)

    # Classification
    elif utils.inspect.isclassifier(model):

        yield datasets.Phishing().take(200)
        yield ((x, np.bool_(y)) for x, y in datasets.Phishing().take(200))

        # Multi-class classification
        if model._multiclass and base.tags.POSITIVE_INPUT not in model._tags:
            yield datasets.ImageSegments().take(200)


def check_learn_one(model, dataset):
    """learn_one should return the calling model and be pure."""

    klass = model.__class__

    for x, y in dataset:

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        model = model.learn_one(x, y)

        # Check the model returns itself
        assert isinstance(model, klass)

        # Check learn_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one(classifier, dataset):
    """predict_proba_one should return a valid probability distribution and be pure."""

    if not hasattr(classifier, "predict_proba_one"):
        return

    for x, y in dataset:

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        classifier = classifier.learn_one(x, y)
        y_pred = classifier.predict_proba_one(x)

        # Check the probabilities are coherent
        assert isinstance(y_pred, dict)
        for proba in y_pred.values():
            assert 0.0 <= proba <= 1.0
        assert math.isclose(sum(y_pred.values()), 1.0)

        # Check predict_proba_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one_binary(classifier, dataset):
    """predict_proba_one should return a dict with True and False keys."""

    for x, y in dataset:
        y_pred = classifier.predict_proba_one(x)
        classifier = classifier.learn_one(x, y)
        assert set(y_pred.keys()) == {False, True}


def check_shuffle_features_no_impact(model, dataset):
    """Changing the order of the features between calls should have no effect on a model."""

    from river import utils

    shuffled = copy.deepcopy(model)

    for x, y in dataset:

        # Shuffle the features
        features = list(x.keys())
        random.shuffle(features)
        x_shuffled = {i: x[i] for i in features}

        assert x == x_shuffled  # order doesn't matter for dicts

        y_pred = model.predict_one(x)
        y_pred_shuffled = shuffled.predict_one(x_shuffled)

        if utils.inspect.ismoregressor(model):
            for o in y_pred:
                assert math.isclose(y_pred[o], y_pred_shuffled[o])
        elif utils.inspect.isregressor(model):
            assert math.isclose(y_pred, y_pred_shuffled)
        else:
            assert y_pred == y_pred_shuffled

        model.learn_one(x, y)
        shuffled.learn_one(x_shuffled, y)


def check_emerging_features(model, dataset):
    """The model should work fine when new features appear."""

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        model.predict_one(x)
        model.learn_one({i: x[i] for i in features[:-3]}, y)  # drop 3 features at random


def check_disappearing_features(model, dataset):
    """The model should work fine when features disappear."""

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        model.predict_one({i: x[i] for i in features[:-3]})  # drop 3 features at random
        model.learn_one(x, y)


def check_debug_one(model, dataset):
    for x, y in dataset:
        model.debug_one(x)
        model.learn_one(x, y)
        model.debug_one(x)
        break


def check_pickling(model, dataset):
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)
    for x, y in dataset:
        model.predict_one(x)
        model.learn_one(x, y)
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)


def check_has_tag(model, tag):
    assert tag in model._tags


def check_repr(model):
    assert isinstance(repr(model), str)


def check_str(model):
    assert isinstance(str(model), str)


def check_tags(model):
    """Checks that the `_tags` property works."""
    assert isinstance(model._tags, set)


def check_set_params_idempotent(model):
    assert len(model.__dict__) == len(model._set_params().__dict__)


def check_init(model):
    params = model._unit_test_params()
    assert isinstance(model.__class__(**params), model.__class__)


def check_doc(model):
    assert model.__doc__


def check_clone(model):
    clone = model.clone()
    assert id(clone) != id(model)
    assert dir(clone) == dir(model)


def wrapped_partial(func, *args, **kwargs):
    """

    Taken from http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/

    """
    partial = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial, func)
    return partial


def allow_exception(func, exception):
    def f(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except exception:
            pass

    f.__name__ = func.__name__
    return f


def yield_checks(model):
    """Generates unit tests for a given model.

    Parameters:
        model (base.Estimator)

    """

    from river import utils

    # General checks
    yield check_repr
    yield check_str
    yield check_tags
    yield check_set_params_idempotent
    yield check_init
    yield check_doc
    yield check_clone

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

    # Classifier checks
    if utils.inspect.isclassifier(model) and not utils.inspect.ismoclassifier(model):
        checks.append(allow_exception(check_predict_proba_one, NotImplementedError))
        # Specific checks for binary classifiers
        if not model._multiclass:
            checks.append(allow_exception(check_predict_proba_one_binary, NotImplementedError))

    for check in checks:
        for dataset in yield_datasets(model):
            yield wrapped_partial(check, dataset=dataset)


def check_estimator(model):
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
