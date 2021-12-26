import copy
import inspect
import pickle
import random

from .utils import assert_predictions_are_close, seed_params


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


def check_shuffle_features_no_impact(model, dataset):
    """Changing the order of the features between calls should have no effect on a model."""

    from river import utils

    params = seed_params(model._get_params(), seed=42)
    model = model._set_params(params)
    shuffled = copy.deepcopy(model)

    for x, y in dataset:

        # Shuffle the features
        features = list(x.keys())
        random.shuffle(features)
        x_shuffled = {i: x[i] for i in features}

        assert x == x_shuffled  # order doesn't matter for dicts

        if utils.inspect.isclassifier(model):
            try:
                y_pred = model.predict_proba_one(x)
                y_pred_shuffled = shuffled.predict_proba_one(x_shuffled)
            except NotImplementedError:
                y_pred = model.predict_one(x)
                y_pred_shuffled = shuffled.predict_one(x_shuffled)
        else:
            y_pred = model.predict_one(x)
            y_pred_shuffled = shuffled.predict_one(x_shuffled)

        assert_predictions_are_close(y_pred, y_pred_shuffled)

        model.learn_one(x, y)
        shuffled.learn_one(x_shuffled, y)


def check_emerging_features(model, dataset):
    """The model should work fine when new features appear."""

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        model.predict_one(x)
        model.learn_one(
            {i: x[i] for i in features[:-3]}, y
        )  # drop 3 features at random


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


def check_init_has_default_params_for_tests(model):
    for params in model._unit_test_params():
        assert isinstance(model.__class__(**params), model.__class__)


def check_init_default_params_are_not_mutable(model):
    """Mutable parameters in signatures are discouraged, as explained in
    https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments

    We enforce immutable parameters by only allowing a certain list of basic types.

    """

    allowed = (type(None), float, int, tuple, str, bool, type)

    for param in inspect.signature(model.__class__).parameters.values():
        assert param.default is inspect._empty or isinstance(param.default, allowed)


def check_doc(model):
    assert model.__doc__


def check_clone(model):
    clone = model.clone()
    assert id(clone) != id(model)
    assert dir(clone) == dir(model)


def check_seeding_is_idempotent(model, dataset):

    params = model._get_params()
    seeded_params = seed_params(params, seed=42)

    A = model._set_params(seeded_params)
    B = model._set_params(seeded_params)

    for x, y in dataset:
        assert A.predict_one(x) == B.predict_one(x)
        A.learn_one(x, y)
        B.learn_one(x, y)
