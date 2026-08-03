from __future__ import annotations

import datetime as dt

import pandas as pd

from river import compose, feature_extraction, linear_model, preprocessing, stats, utils
from river.compose.pipeline import _method_params, _route


def test_pipeline_funcs():
    def a(x):
        pass

    def b(x):
        pass

    pipelines = [
        compose.FuncTransformer(a) | b,
        compose.FuncTransformer(a) | ("b", b),
        compose.FuncTransformer(a) | ("b", compose.FuncTransformer(b)),
        a | compose.FuncTransformer(b),
        ("a", a) | compose.FuncTransformer(b),
        ("a", compose.FuncTransformer(a)) | compose.FuncTransformer(b),
    ]

    for pipeline in pipelines:
        assert str(pipeline) == "a | b"


def test_pipeline_add_at_start():
    def a(x):
        pass

    pipeline = preprocessing.StandardScaler() | linear_model.LinearRegression()
    pipeline = a | pipeline
    assert str(pipeline) == "a | StandardScaler | LinearRegression"


def test_union_funcs():
    def a(x):
        pass

    def b(x):
        pass

    pipelines = [
        compose.FuncTransformer(a) + b,
        compose.FuncTransformer(a) + ("b", b),
        compose.FuncTransformer(a) + ("b", compose.FuncTransformer(b)),
        a + compose.FuncTransformer(b),
        ("a", a) + compose.FuncTransformer(b),
        ("a", compose.FuncTransformer(a)) + compose.FuncTransformer(b),
    ]

    for i, pipeline in enumerate(pipelines):
        print(i, str(pipeline))
        assert str(pipeline) == "a + b"


def test_learn_one_with_learn_during_predict():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), bool(x % 2)) for x in range(100)]

    for x, y in dataset:
        counts_pre = dict(pipeline["scale"].counts)
        with compose.learn_during_predict():
            pipeline.learn_one(x, y)
        counts_no_learn = dict(pipeline["scale"].counts)
        pipeline.learn_one(x, y)
        counts_post = dict(pipeline["scale"].counts)

        assert counts_pre != counts_post
        assert counts_pre == counts_no_learn


def test_learn_many_with_learn_during_predict():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for i in range(0, len(dataset), 5):
        X = pd.DataFrame([x for x, _ in dataset][i : i + 5])
        y = pd.Series([bool(y % 2) for _, y in dataset][i : i + 5])

        counts_pre = dict(pipeline["scale"].counts)
        with compose.learn_during_predict():
            pipeline.learn_many(X, y)
        counts_no_learn = dict(pipeline["scale"].counts)
        pipeline.learn_many(X, y)
        counts_post = dict(pipeline["scale"].counts)

        assert counts_pre != counts_post
        assert counts_pre == counts_no_learn


def test_list_of_funcs():
    def f(x):
        return {"f": 1}

    def g(x):
        return {"g": 2}

    def times_2(x):
        return {k: v * 2 for k, v in x.items()}

    expected = {"f": 2, "g": 4}
    assert compose.Pipeline([f, g], times_2).transform_one(None) == expected
    assert ([f, g] | compose.FuncTransformer(times_2)).transform_one(None) == expected


def test_get():
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    assert isinstance(model["StandardScaler"], preprocessing.StandardScaler)
    assert isinstance(model["LinearRegression"], linear_model.LinearRegression)
    assert isinstance(model[0], preprocessing.StandardScaler)
    assert isinstance(model[1], linear_model.LinearRegression)
    assert isinstance(model[-1], linear_model.LinearRegression)
    assert isinstance(model[-2], preprocessing.StandardScaler)


def test_learn_one_routes_extra_params_to_supervised_transformer():
    """Extra keyword arguments passed to ``learn_one`` (such as a timestamp ``t``) should be
    forwarded to the steps whose ``learn_one`` accepts them, and silently dropped for the steps
    that don't (e.g. the final model). See https://github.com/online-ml/river/issues/1600.

    """
    model = (
        feature_extraction.TargetAgg(
            by="group", how=utils.TimeRolling(stats.Mean, dt.timedelta(days=7))
        )
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression()
    )

    t0 = dt.datetime(2023, 1, 1)
    for day in range(30):
        x = {"group": "a"}
        y = bool(day % 2)
        t = t0 + dt.timedelta(days=day)
        model.predict_one(x)
        model.learn_one(x, y, t=t)

    # The timestamp must have reached the TimeRolling-backed aggregate.
    assert model["TargetAgg"]._groups[("a",)].get() is not None


def test_learn_one_routes_extra_params_to_unsupervised_transformer_in_union():
    """An unsupervised ``feature_extraction.Agg`` nested in a ``TransformerUnion`` should also
    receive a routed ``t``."""
    agg = feature_extraction.Agg(
        on="v", by="group", how=utils.TimeRolling(stats.Mean, dt.timedelta(days=7))
    )
    model = (agg + compose.Select("v")) | linear_model.LinearRegression()

    t0 = dt.datetime(2023, 1, 1)
    for day in range(30):
        x = {"group": "a", "v": float(day)}
        model.learn_one(x, float(day), t=t0 + dt.timedelta(days=day))

    assert model["TransformerUnion"]["Agg"]._groups[("a",)].get() is not None


def test_learn_one_without_extra_params_is_unaffected():
    """A pipeline with no extra ``learn_one`` arguments behaves exactly as before."""
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    for i in range(10):
        model.learn_one({"a": float(i)}, float(i))
    assert model.predict_one({"a": 1.0}) is not None


def test_learn_one_rejects_unknown_param_for_final_estimator():
    """A genuinely unknown argument that no step accepts is dropped rather than raising; the
    final estimator only receives what its ``learn_one`` declares."""
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    # ``t`` is accepted by no step here, so it must be silently ignored everywhere.
    model.learn_one({"a": 1.0}, 1.0, t=dt.datetime(2023, 1, 1))


class _RecordingRegressor(linear_model.LinearRegression):
    """A regressor whose ``learn_one`` swallows arbitrary keyword arguments, recording them."""

    def learn_one(self, x, y, **kwargs):
        self.received = kwargs
        super().learn_one(x, y)


def test_learn_one_forwards_all_params_to_kwargs_accepting_step():
    """A step whose ``learn_one`` declares ``**kwargs`` receives every extra argument."""
    final = _RecordingRegressor()
    model = preprocessing.StandardScaler() | final
    model.learn_one({"a": 1.0}, 1.0, t=dt.datetime(2023, 1, 1), w=2.0)
    assert final.received == {"t": dt.datetime(2023, 1, 1), "w": 2.0}


def test_predict_one_routes_extra_params_during_learn_during_predict():
    """In ``learn_during_predict`` mode, ``predict_one(x, t=...)`` must route ``t`` to the
    unsupervised ``TimeRolling``-backed step learning during predict, and must not leak it into
    the final estimator's ``predict_one``."""
    agg = feature_extraction.Agg(
        on="v", by="g", how=utils.TimeRolling(stats.Mean, dt.timedelta(days=7))
    )
    model = (agg + compose.Select("v")) | linear_model.LinearRegression()

    t0 = dt.datetime(2023, 1, 1)
    with compose.learn_during_predict():
        for day in range(30):
            x = {"g": "a", "v": float(day)}
            model.predict_one(x, t=t0 + dt.timedelta(days=day))
            model.learn_one(x, float(day), t=t0 + dt.timedelta(days=day))

    assert model["TransformerUnion"]["Agg"]._groups[("a",)].get() is not None


def test_method_params_blueprint():
    """`_method_params` reports the routable arguments per step and method."""
    assert _method_params(preprocessing.StandardScaler(), "learn_one") == frozenset()
    # LinearRegression.learn_one(self, x, y, w=1.0) exposes a routable sample weight.
    assert _method_params(linear_model.LinearRegression(), "learn_one") == frozenset({"w"})
    assert _method_params(
        feature_extraction.Agg(on="v", by="g", how=stats.Mean()), "learn_one"
    ) == frozenset({"t"})
    assert _method_params(
        feature_extraction.TargetAgg(by="g", how=stats.Mean()), "learn_one"
    ) == frozenset({"t"})
    # A ``**kwargs`` signature routes everything; an absent method routes nothing extra.
    assert _method_params(_RecordingRegressor(), "learn_one") is None
    assert _method_params(linear_model.LinearRegression(), "predict_one") == frozenset()


def test_route_blueprint():
    """`_route` keeps only the accepted subset, with `None` meaning 'forward everything'."""
    params = {"t": 1, "w": 2}
    assert _route(frozenset(), params) == {}
    assert _route(frozenset({"t"}), params) == {"t": 1}
    assert _route(None, params) is params  # forward all, no copy
    assert _route(frozenset({"t"}), {}) == {}  # empty params short-circuits
