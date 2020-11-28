"""Utilities for unit testing and sanity checking estimators."""
import copy
import functools
import math
import pickle
import random

import numpy as np


__all__ = ['check_estimator']


def yield_datasets(model):

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
                oh = (compose.SelectType(str) | preprocessing.OneHotEncoder()) + compose.SelectType(int)
                for x, y in datasets.SolarFlare():
                    yield oh.transform_one(x), y
        yield SolarFlare()

    # Regression
    elif utils.inspect.isregressor(model):
        yield datasets.TrumpApproval()

    # Multi-output classification
    if utils.inspect.ismoclassifier(model):
        yield datasets.Music()

    # Classification
    elif utils.inspect.isclassifier(model):

        yield datasets.Phishing()
        yield ((x, np.bool_(y)) for x, y in datasets.Phishing())

        # Multi-class classification
        if model._multiclass:
            yield datasets.ImageSegments().take(500)


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

    for x, y in dataset:

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        classifier = classifier.learn_one(x, y)
        y_pred = classifier.predict_proba_one(x)

        # Check the probabilities are coherent
        assert isinstance(y_pred, dict)
        assert math.isclose(sum(y_pred.values()), 1.)
        for proba in y_pred.values():
            assert 0. <= proba <= 1.

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


def check_predict_emerging_feature(model, dataset):
    """predict_one should work even with previously unseen features."""

    x, y = next(iter(dataset))
    features = list(x.keys())[:-1]

    model.learn_one({i: x[i] for i in features}, y)
    model.predict_one(x)


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
    rep = repr(model)
    assert isinstance(rep, str)


def check_str(model):
    assert isinstance(str(model), str)


def check_tags(model):
    """Checks that the `_tags` property works."""
    assert isinstance(model._tags, set)


def check_set_params_idempotent(model):
    assert len(model.__dict__) == len(model._set_params().__dict__)


def check_init(model):
    try:
        params = model._default_params()
    except AttributeError:
        params = {}
    assert isinstance(model.__class__(**params), model.__class__)


def check_doc(model):
    assert model.__doc__


def wrapped_partial(func, *args, **kwargs):
    """

    Taken from http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/

    """
    partial = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial, func)
    return partial


def with_ignore_exception(func, exception):
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

    # Checks that make use of datasets
    for dataset in yield_datasets(model):

        yield wrapped_partial(check_learn_one, dataset=dataset)
        yield wrapped_partial(check_pickling, dataset=dataset)
        yield wrapped_partial(check_shuffle_features_no_impact, dataset=dataset)
        yield wrapped_partial(check_predict_emerging_feature, dataset=dataset)

        if hasattr(model, 'debug_one'):
            yield wrapped_partial(check_debug_one, dataset=dataset)

        # Classifier checks
        if utils.inspect.isclassifier(model) and not utils.inspect.ismoclassifier(model):

            # Some classifiers do not implement predict_proba_one
            yield with_ignore_exception(
                wrapped_partial(check_predict_proba_one, dataset=dataset),
                NotImplementedError
            )

            # Specific checks for binary classifiers
            if not model._multiclass:
                yield with_ignore_exception(
                    wrapped_partial(check_predict_proba_one_binary, dataset=dataset),
                    NotImplementedError
                )


def check_estimator(model):
    """Check if a model adheres to `river`'s conventions.

    This will run a series of unit tests. The nature of the unit tests depends on the type of
    model.

    Parameters
    ----------
    model

    """
    for check in yield_checks(model):
        check(copy.deepcopy(model))
