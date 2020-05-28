"""Utilities for unit testing and sanity checking estimators."""
import copy
import functools
import inspect
import itertools
import math
import pickle


__all__ = [
    'check_estimator',
    'guess_model'
]


def guess_model(model):
    """Extracts the relevant part model.

    Sometimes we need to check if a model can perform regression, classification, etc. When the
    model is a pipeline, then this can be checked by looking at the final step of the pipeline.

    This function is highly succeptible to be modified or disappear altogether. The only reason it
    exists is that we can't do `isinstance(pipeline, base.Regressor)`.

    Parameters:
        model

    """

    from creme import base
    from creme import compose

    if isinstance(model, (base.MultiOutputRegressor, base.MultiOutputClassifier)):
        return model
    elif isinstance(model, compose.Pipeline):
        return guess_model(list(model.steps.values())[-1])
    elif isinstance(model, base.Wrapper):
        return guess_model(model._wrapped_model)
    return model


def yield_datasets(model):

    from creme import base
    from creme import datasets
    from creme import stream
    from sklearn import datasets as sk_datasets

    model = guess_model(model)

    if isinstance(model, (base.BinaryClassifier, base.MultiClassifier)):
        yield datasets.Phishing()
    if isinstance(model, base.MultiClassifier):
        yield datasets.ImageSegments().take(500)
    if isinstance(model, base.Regressor):
        yield datasets.TrumpApproval()
    if isinstance(model, base.MultiOutputRegressor):
        yield stream.iter_sklearn_dataset(sk_datasets.load_linnerud())
    if isinstance(model, base.MultiOutputClassifier):
        yield datasets.Music()


def check_fit_one(model, dataset):

    klass = model.__class__

    for x, y in dataset:

        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        model = model.fit_one(x, y)

        # Check the model returns itself
        assert isinstance(model, klass)

        # Check fit_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_predict_proba_one(classifier, dataset):

    for x, y in dataset:

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


def check_predict_proba_one_binary(classifier, dataset):

    for x, y in dataset:
        y_pred = classifier.predict_proba_one(x)
        classifier = classifier.fit_one(x, y)
        assert len(y_pred) == 2
        assert True in y_pred
        assert False in y_pred


def check_debug_one(model, dataset):
    for x, y in dataset:
        model.debug_one(x)
        model.fit_one(x, y)
        model.debug_one(x)
        break


def check_pickling(model, dataset):
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)
    for x, y in dataset:
        model.predict_one(x)
        model.fit_one(x, y)
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)


def check_repr_works(model):
    rep = repr(model)
    assert isinstance(rep, str)


def check_str_works(model):
    assert isinstance(str(model), str)


def check_tags(model):
    """Checks that the `_tags` property works."""
    assert isinstance(model._tags, dict)


def check_set_params_idempotent(model):
    assert len(model.__dict__) == len(model._set_params().__dict__)


def yield_checks(model):
    """Generates unit tests that can be applied to a given model.

    Parameters:
        model (base.Estimator)

    """

    from creme import base

    yield check_repr_works
    yield check_str_works
    yield check_tags
    yield check_set_params_idempotent

    for dataset in yield_datasets(model):

        def with_dataset(method):
            check = functools.partial(check_fit_one, dataset=dataset)
            functools.update_wrapper(check, method)
            return check

        yield with_dataset(check_fit_one)
        yield with_dataset(check_pickling)
        if hasattr(model, 'debug_one'):
            yield with_dataset(check_debug_one)

        model = guess_model(model)

        if isinstance(model, base.Classifier):
            yield with_dataset(check_predict_proba_one)

        if not isinstance(model, base.MultiClassifier):  # only binary classifiers
            yield with_dataset(check_predict_proba_one_binary)


def check_estimator(model):
    """Check if a model adheres to `creme`'s conventions.

    Parameters:
        model

    """
    for check in yield_checks(model):
        check(copy.deepcopy(model))
