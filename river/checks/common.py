from __future__ import annotations

import copy
import gc
import inspect
import itertools
import math
import pickle
import random

from .utils import assert_predictions_are_close, seed_params


def _inference_methods(model):
    """Yield the names of inference methods this model exposes."""

    from river import base
    from river.anomaly.base import AnomalyDetector

    if isinstance(model, AnomalyDetector):
        yield "score_one"
    elif isinstance(model, base.Classifier):
        if not isinstance(model, base.MultiLabelClassifier):
            yield "predict_proba_one"
        yield "predict_one"
    elif isinstance(model, base.Regressor):
        yield "predict_one"
    elif isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        yield "transform_one"


def _infer(model, x):
    """Call the model's primary inference method on `x`."""

    from river import base
    from river.anomaly.base import AnomalyDetector

    if isinstance(model, AnomalyDetector):
        return model.score_one(x)
    if isinstance(model, base.Classifier):
        try:
            return model.predict_proba_one(x)
        except NotImplementedError:
            return model.predict_one(x)
    if isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        return model.transform_one(x)
    return model.predict_one(x)


def _learn(model, x, y):
    """Call `learn_one` with the right signature for the model's kind."""

    from river.anomaly.base import AnomalyDetector

    if isinstance(model, AnomalyDetector):
        model.learn_one(x)
    elif model._supervised:
        model.learn_one(x, y)
    else:
        model.learn_one(x)


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

    from river.active.base import ActiveLearningClassifier

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)
    shuffled = copy.deepcopy(model)

    for x, y in dataset:
        # Shuffle the features
        features = list(x.keys())
        random.shuffle(features)
        x_shuffled = {i: x[i] for i in features}

        assert x == x_shuffled  # order doesn't matter for dicts

        y_pred = _infer(model, x)
        y_pred_shuffled = _infer(shuffled, x_shuffled)

        if isinstance(model, ActiveLearningClassifier):
            y_pred, _ = y_pred
            y_pred_shuffled, _ = y_pred_shuffled

        assert_predictions_are_close(y_pred, y_pred_shuffled)

        _learn(model, x, y)
        _learn(shuffled, x_shuffled, y)


def check_predict_one_pure(model, dataset):
    """Inference methods should not mutate their inputs.

    Covers `predict_one`, `predict_proba_one`, `score_one`, and `transform_one`
    depending on the estimator's type. The `learn_one` counterpart is already
    covered by `check_learn_one`.
    """

    methods = list(_inference_methods(model))
    if not methods:
        return

    for x, y in dataset:
        # Learn first so estimators that need state (e.g. running statistics)
        # have something to work with. Cold-start behaviour is covered by
        # check_predict_one_before_any_learn.
        _learn(model, x, y)
        for name in methods:
            method = getattr(model, name, None)
            if method is None:
                continue
            xx = copy.deepcopy(x)
            try:
                method(x)
            except NotImplementedError:
                continue
            assert x == xx, f"{name} mutated its input"


def check_emerging_features(model, dataset):
    """The model should work fine when new features appear."""

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        _infer(model, x)
        _learn(model, {i: x[i] for i in features[:-3]}, y)  # drop 3 features at random


def check_disappearing_features(model, dataset):
    """The model should work fine when features disappear."""

    for x, y in dataset:
        features = list(x.keys())
        random.shuffle(features)
        _infer(model, {i: x[i] for i in features[:-3]})  # drop 3 features at random
        _learn(model, x, y)


def check_radically_disappearing_features(model, dataset):
    """The model should work fine when nearly all features disappear."""

    # First give all the data to prime the model
    for x, y in itertools.islice(dataset, 20):
        _infer(model, x)
        _learn(model, x, y)

    # And suddenly remove almost everything
    for x, y in itertools.islice(dataset, 10, None):
        features = list(x.keys())
        feat = random.choice(list(features))  # keep only 1 feature, at random
        _infer(model, {feat: x[feat]})
        _learn(model, {feat: x[feat]}, y)


def check_debug_one(model, dataset):
    for x, y in dataset:
        model.debug_one(x)
        if model._supervised:
            model.learn_one(x, y)
        else:
            model.learn_one(x)
        model.debug_one(x)
        break


def check_bounded_memory_growth(model, dataset):
    """A truly online model's memory should not grow unboundedly with samples.

    The dataset is split in half: a warmup phase (during which the model is
    allowed to grow freely as features are discovered, dicts rehash, buffers
    fill, etc.) and an equally sized measurement phase. The measurement-phase
    growth is then compared against the warmup-phase growth. A genuinely
    online model's growth slows after warmup, so measurement-phase growth
    should be at most comparable to warmup-phase growth. A model that retains
    state per sample would keep growing at the same rate and blow past that.

    This is intentionally lenient: the goal is to catch dramatic regressions
    (e.g. accidentally storing every sample) on heterogeneous CI machines, not
    to police every byte. Measurement is noisy in practice — Python's garbage
    collector, dict capacity jumps, and bursty tree splits can all shift the
    numbers by tens of percent between runs.
    """

    samples = list(dataset)
    if len(samples) < 4:
        return  # not enough data to split meaningfully

    warmup_end = len(samples) // 2

    for x, y in samples[:warmup_end]:
        _learn(model, x, y)
    gc.collect()
    size_after_warmup = model._raw_memory_usage

    for x, y in samples[warmup_end:]:
        _learn(model, x, y)
    gc.collect()
    size_final = model._raw_memory_usage

    warmup_growth = size_after_warmup  # baseline is an empty model (~0 B)
    measurement_growth = size_final - size_after_warmup
    # Allow the measurement phase to grow by up to 3× the warmup phase plus
    # a 16 KiB absolute floor. Same number of samples in each phase, so a
    # model growing at a constant per-sample rate would land near 1×; bursty
    # tree-ensemble splits (whose timing depends on data order and may all
    # land in the measurement phase) can push that ratio higher. 3× catches
    # dramatic acceleration without flagging benign bursts.
    tolerance = max(16 * 1024, warmup_growth // 4)
    limit = 3 * max(warmup_growth, 0) + tolerance

    assert measurement_growth <= limit, (
        f"Model memory grew unboundedly: "
        f"{size_after_warmup} B after {warmup_end} warmup samples -> "
        f"{size_final} B after {len(samples)} samples. "
        f"Measurement-phase growth {measurement_growth:+d} B exceeds "
        f"the {limit} B limit (3× warmup growth {warmup_growth} B + "
        f"{tolerance} B tolerance)."
    )


def check_pickling_supports_roundtrip(model):
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)


def check_pickling(model, dataset):
    assert isinstance(pickle.loads(pickle.dumps(model)), model.__class__)
    for x, y in dataset:
        _infer(model, x)
        _learn(model, x, y)
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


def check_repr_roundtrips_clone(model):
    """`clone` must preserve every `__init__` parameter, which `repr` exposes."""
    assert repr(model) == repr(model.clone())


def check_clone_with_new_params_applies(model):
    """`clone(new_params={p: v})` must actually apply the given values.

    We try each eligible scalar parameter independently — if the constructor
    rejects the new value (e.g. because of a cross-parameter invariant), we skip
    that parameter. Some estimators wrap scalar parameters into rich objects
    (e.g. `Constant(lr)`), so we accept either an exact match with the new
    value or strict inequality with the original attribute as evidence that
    the override took effect.
    """

    for name, param in inspect.signature(model.__class__).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = param.default
        if isinstance(default, bool):
            new_value = not default
        elif isinstance(default, int) and not isinstance(default, bool):
            new_value = default + 1
        elif isinstance(default, float):
            new_value = default + 1.0
        else:
            continue
        try:
            clone = model.clone(new_params={name: new_value})
        except Exception:
            continue
        original_attr = getattr(model, name, default)
        clone_attr = getattr(clone, name)
        assert clone_attr == new_value or clone_attr != original_attr, (
            f"clone did not apply new param {name!r}: "
            f"expected {new_value!r}, got {clone_attr!r} (original was {original_attr!r})"
        )


def check_get_params_matches_signature(model):
    """`_get_params()` must expose every keyword in the `__init__` signature.

    Catches estimators that accept a parameter in `__init__` but don't store
    it as a same-named attribute — which would silently break `clone`.
    """

    expected = {
        name
        for name, param in inspect.signature(model.__class__).parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }
    params = model._get_params()
    actual = {k for k in params.keys() if k != "_POSITIONAL_ARGS"}
    missing = expected - actual
    assert not missing, f"_get_params() is missing entries from __init__: {sorted(missing)}"


def check_clone_is_independent(model, dataset):
    """Training the original after cloning must not affect the clone.

    `check_clone_changes_memory_addresses` only checks the top-level object ids
    differ. This check guards against deeper shared state (e.g. a list of base
    models stored by reference).
    """

    clone = model.clone()
    snapshot = pickle.dumps(clone)

    for x, y in itertools.islice(dataset, 30):
        _learn(model, x, y)

    assert pickle.dumps(clone) == snapshot, (
        "training the original mutated the clone — clone() likely shared mutable state"
    )


def check_predict_one_before_any_learn(model, dataset):
    """Inference on a fresh estimator must work (or raise `NotImplementedError`).

    River's online-learning convention is that `predict_one` / `score_one` /
    `transform_one` are callable from the very first event, before any
    `learn_one` has been issued.
    """

    methods = list(_inference_methods(model))
    if not methods:
        return

    for x, _ in dataset:
        for name in methods:
            method = getattr(model, name, None)
            if method is None:
                continue
            try:
                method(x)
            except NotImplementedError:
                continue
        break


def check_no_state_aliasing_with_input(model, dataset):
    """Mutating `x` after `learn_one` must not change the model's state.

    Catches estimators that stash a reference to the input dict instead of
    copying the values they care about.
    """

    for x, y in dataset:
        x_copy = copy.deepcopy(x)
        _learn(model, x_copy, copy.deepcopy(y))

        before = pickle.dumps(model)

        # Mutate x_copy in place: change numeric values, drop a key.
        for k in list(x_copy.keys()):
            v = x_copy[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                x_copy[k] = v + 1e9
            elif isinstance(v, str):
                x_copy[k] = v + "_mutated"
        if x_copy:
            x_copy.pop(next(iter(x_copy)))

        assert pickle.dumps(model) == before, (
            "model state changed after mutating x — learn_one stored a reference instead of copying"
        )
        break


def check_transform_one(model, dataset):
    """`transform_one` should return a dict and be reachable from the public API."""

    from river import base

    if not isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        return

    for x, y in dataset:
        if isinstance(model, base.SupervisedTransformer):
            model.learn_one(x, y)
        else:
            model.learn_one(x)
        out = model.transform_one(x)
        assert isinstance(out, dict), f"transform_one returned {type(out).__name__}, expected dict"


def _assert_dict_predictions_match(a, b, tolerance=1e-5):
    import math

    keys = set(a) | set(b)
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        if isinstance(va, float) or isinstance(vb, float):
            assert math.isclose(float(va), float(vb), rel_tol=tolerance, abs_tol=tolerance), (
                f"key {k!r}: {va} != {vb}"
            )
        else:
            assert va == vb, f"key {k!r}: {va} != {vb}"


def _train_and_split(model, dataset, n_train=30, n_test=20):
    """Train `model` with `learn_one` on the first slice; return the held-out slice."""

    rows = list(itertools.islice(dataset, n_train + n_test))
    train, test = rows[:n_train], rows[n_train:]
    for x, y in train:
        _learn(model, x, y)
    return test


def check_predict_many_matches_predict_one(model, dataset):
    """`predict_many` on a batch must agree with `predict_one` per row."""

    import pandas as pd

    from river import base

    if not isinstance(model, (base.MiniBatchClassifier, base.MiniBatchRegressor)):
        return

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)
    test = _train_and_split(model, dataset)
    if not test:
        return

    X = pd.DataFrame([x for x, _ in test])
    try:
        many = model.predict_many(X).tolist()
    except (AttributeError, NotImplementedError):
        # Mixed pipelines may declare themselves MiniBatchClassifier/Regressor
        # but contain a step that lacks transform_many/predict_many.
        return
    one = [model.predict_one(x) for x, _ in test]

    for m, o in zip(many, one):
        if m is None or o is None:
            assert m == o
        elif isinstance(model, base.Regressor):
            assert_predictions_are_close(float(m), float(o))
        else:
            assert m == o, f"predict_many vs predict_one disagree: {m!r} != {o!r}"


def check_learn_many_matches_learn_one(model, dataset):
    """`learn_many` over a batch must match a `learn_one` loop, even as features come and go.

    Applies to every mini-batch estimator — regressors, classifiers and transformers — and
    compares whatever the model produces per row (`predict_one`, `predict_proba_one` or
    `transform_one`). Estimators whose mini-batch update is not equivalent to the per-row update
    declare this check in `_unit_test_skips`: the gradient-descent linear models (one
    mean-gradient step per batch rather than one step per row) and naive Bayes (whose
    `learn_many` consumes sparse count matrices, covered by its own test).
    """

    import pandas as pd

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)

    rows = list(itertools.islice(dataset, 60))
    features = list(rows[0][0]) if rows else []
    if len(rows) < 25 or len(features) < 2:
        return

    # Feature subsets that emerge, disappear, reappear, and radically shrink across batches, so
    # the equivalence is exercised under a changing feature space rather than a fixed schema.
    half = len(features) // 2
    subsets = [
        features[:half],  # first half only
        features,  # second half emerges
        features[half:],  # first half disappears
        features[:1],  # all but one disappear
        features,  # everything reappears
    ]
    per = len(rows) // len(subsets)

    supervised = model._supervised
    one, many = model.clone(), model.clone()
    queries = []
    for i, cols in enumerate(subsets):
        chunk = rows[i * per : (i + 1) * per]
        batch = [{c: float(x[c]) for c in cols} for x, _ in chunk]
        targets = [y for _, y in chunk]
        for x, y in zip(batch, targets):
            _learn(one, x, y)
        X = pd.DataFrame(batch, columns=cols)
        try:
            many.learn_many(X, pd.Series(targets)) if supervised else many.learn_many(X)
        except (AttributeError, NotImplementedError):
            # A pipeline/union may declare itself mini-batch yet wrap a step without learn_many.
            return
        queries.extend(batch)

    # `learn_one` and `learn_many` accumulate the same state but may differ at the floating-point
    # level (e.g. a chained rank-1 inverse update versus a single inverse), so this is looser than
    # `assert_predictions_are_close`; a real discrepancy is far larger.
    def assert_close(a, b):
        if isinstance(a, dict):  # predict_proba_one / transform_one
            _assert_dict_predictions_match(a, b, tolerance=1e-4)
        elif isinstance(a, float):  # predict_one for a regressor
            assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)
        else:  # a class label
            assert a == b

    for x in queries:
        assert_close(_infer(one, x), _infer(many, x))


def check_predict_proba_many_matches_predict_proba_one(model, dataset):
    """`predict_proba_many` on a batch must agree with `predict_proba_one` per row."""

    import pandas as pd

    from river import base

    if not isinstance(model, base.MiniBatchClassifier):
        return

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)
    test = _train_and_split(model, dataset)
    if not test:
        return

    X = pd.DataFrame([x for x, _ in test])
    try:
        many = model.predict_proba_many(X)
    except (AttributeError, NotImplementedError):
        return

    for i, (x, _) in enumerate(test):
        try:
            one = model.predict_proba_one(x)
        except NotImplementedError:
            return
        many_row = {k: many.iloc[i][k] for k in many.columns}
        _assert_dict_predictions_match(one, many_row)


def check_transform_many_matches_transform_one(model, dataset):
    """`transform_many` on a batch must agree with `transform_one` per row."""

    import pandas as pd

    from river import base

    if not isinstance(model, (base.MiniBatchTransformer, base.MiniBatchSupervisedTransformer)):
        return

    params = seed_params(model._get_params(), seed=42)
    model = model.clone(params)
    test = _train_and_split(model, dataset)
    if not test:
        return

    X = pd.DataFrame([x for x, _ in test])
    try:
        many = model.transform_many(X)
    except (AttributeError, NotImplementedError):
        return
    for i, (x, _) in enumerate(test):
        one = model.transform_one(x)
        many_row = many.iloc[i].to_dict()
        _assert_dict_predictions_match(one, many_row)


def check_seeding_is_idempotent(model, dataset):
    params = model._get_params()
    seeded_params = seed_params(params, seed=42)

    A = model.clone(seeded_params)
    B = model.clone(seeded_params)

    for x, y in dataset:
        assert _infer(A, x) == _infer(B, x)
        _learn(A, x, y)
        _learn(B, x, y)


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
