from __future__ import annotations

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

        if model._supervised:
            model.learn_one(x, y)
        else:
            model.learn_one(x)

        # Check the model returns itself
        assert isinstance(model, klass)

        # Check learn_one is pure (i.e. x and y haven't changed)
        assert x == xx
        assert y == yy


def check_shuffle_features_no_impact(model, dataset):
    """Changing the order of the features between calls should have no effect on a model."""

    from river import utils

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)
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
        elif utils.inspect.isanomalydetector(model):
            y_pred = model.score_one(x)
            y_pred_shuffled = shuffled.score_one(x_shuffled)
        else:
            y_pred = model.predict_one(x)
            y_pred_shuffled = shuffled.predict_one(x_shuffled)

        if utils.inspect.isactivelearner(model):
            y_pred, _ = y_pred
            y_pred_shuffled, _ = y_pred_shuffled

        assert_predictions_are_close(y_pred, y_pred_shuffled)

        if utils.inspect.isanomalydetector(model):
            model.learn_one(x)
            shuffled.learn_one(x_shuffled)
        else:
            model.learn_one(x, y)
            shuffled.learn_one(x_shuffled, y)


def check_emerging_features(model, dataset):
    """The model should work fine when new features appear."""
    from river import utils

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        if utils.inspect.isanomalydetector(model):
            model.score_one(x)
        else:
            model.predict_one(x)
        if utils.inspect.isanomalydetector(model):
            model.learn_one({i: x[i] for i in features[:-3]})  # drop 3 features at random
        else:
            model.learn_one({i: x[i] for i in features[:-3]}, y)


def check_disappearing_features(model, dataset):
    """The model should work fine when features disappear."""

    from river import utils

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        if utils.inspect.isanomalydetector(model):
            model.score_one({i: x[i] for i in features[:-3]})  # drop 3 features at random
            model.learn_one(x)
        else:
            model.predict_one({i: x[i] for i in features[:-3]})
            model.learn_one(x, y)


def check_debug_one(model, dataset):
    for x, y in dataset:
        model.debug_one(x)
        if model._supervised:
            model.learn_one(x, y)
        else:
            model.learn_one(x)
        model.debug_one(x)
        break


def check_pickling(model, dataset):
    from river import utils

    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)
    for x, y in dataset:
        if utils.inspect.isanomalydetector(model):
            model.score_one(x)
            model.learn_one(x)
        else:
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


def check_clone_same_class(model):
    assert isinstance(model.clone(), model.__class__)


def check_clone_is_idempotent(model):
    before = model._get_params()
    after = model.clone()._get_params()
    assert len(before) == len(after)
    if isinstance(before, dict):
        assert len(before.keys()) == len(after.keys())
    else:
        assert before == after


def check_mutate_can_be_idempotent(model):
    before = model._get_params()
    model.mutate({})
    after = model._get_params()
    assert before == after


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


def check_clone_changes_memory_addresses(model):
    clone = model.clone()
    assert id(clone) != id(model)
    assert dir(clone) == dir(model)


def check_seeding_is_idempotent(model, dataset):
    from river import utils

    params = model._get_params()
    seeded_params = seed_params(params, seed=42)

    A = model.clone(seeded_params)
    B = model.clone(seeded_params)

    for x, y in dataset:
        if utils.inspect.isanomalydetector(model):
            assert A.score_one(x) == B.score_one(x)
        else:
            assert A.predict_one(x) == B.predict_one(x)
        if model._supervised:
            A.learn_one(x, y)
            B.learn_one(x, y)
        else:
            A.learn_one(x)
            B.learn_one(x)


def check_mutable_attributes_exist(model):
    for attr in model._mutable_attributes:
        assert hasattr(model, attr)


def check_wrapper_accepts_kwargs(wrapper):
    """Check that the wrapper accepts keyword arguments in its methods."""
    for method_name in [
        "score_one",
        "predict_one",
        "predict_proba_one",
        "forecast",
        "learn_one",
        "learn_many",
        "predict_many",
        "predict_proba_many",
    ]:
        if method := getattr(wrapper, method_name, None):
            try:
                method(None)
            except NotImplementedError:
                continue
            except Exception:
                pass
            assert inspect.getfullargspec(method).varkw is not None
