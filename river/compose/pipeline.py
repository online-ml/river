from __future__ import annotations

import collections
import contextlib
import functools
import io
import itertools
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from river import base

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries

    _Features = dict[base.typing.FeatureName, Any]
    _Params = dict[str, object]
    _AcceptedParams = frozenset[str] | None
    _Learn = Callable[..., None]
    _TransformOne = Callable[[_Features], _Features]
    _TransformMany = Callable[[IntoDataFrame], IntoDataFrame]
    _OneAnomalyPlan = tuple[
        typing.Literal["anomaly"],
        base.AnomalyFilter,
        bool,
        _Learn,
        _AcceptedParams,
    ]
    _OneTransformerPlan = tuple[
        typing.Literal["transformer"],
        base.Transformer,
        _TransformOne,
        _Learn | None,
        _Learn | None,
        _AcceptedParams,
        bool,
    ]
    _OneFinalPlan = tuple[
        typing.Literal["final"],
        base.Estimator,
        bool,
        _Learn | None,
        _AcceptedParams,
    ]
    _OnePlan = _OneAnomalyPlan | _OneTransformerPlan | _OneFinalPlan
    _ManyAnomalyPlan = tuple[typing.Literal["anomaly"], base.AnomalyFilter]
    _ManyTransformerPlan = tuple[
        typing.Literal["transformer"],
        base.Transformer,
        _TransformMany | None,
        _Learn | None,
        _Learn | None,
        _AcceptedParams,
        bool,
    ]
    _ManyFinalPlan = tuple[
        typing.Literal["final"],
        base.Estimator,
        bool,
        _Learn | None,
        _AcceptedParams,
    ]
    _ManyPlan = _ManyAnomalyPlan | _ManyTransformerPlan | _ManyFinalPlan

    class _PredictOneFunc(typing.Protocol):
        def __call__(self, x: _Features, **params: object) -> Any: ...

    class _PredictProbaOneFunc(typing.Protocol):
        def __call__(
            self, x: _Features, **params: object
        ) -> dict[base.typing.ClfTarget, float]: ...

    class _ScoreOneFunc(typing.Protocol):
        def __call__(self, x: _Features, **params: object) -> float: ...

    class _PredictManyFunc(typing.Protocol):
        def __call__(self, X: IntoDataFrame) -> IntoSeries: ...

    class _PredictProbaManyFunc(typing.Protocol):
        def __call__(self, X: IntoDataFrame) -> IntoDataFrame: ...


from . import func, grouper, product, union
from ._routing import method_params as _method_params
from ._routing import route as _route

__all__ = ["Pipeline"]


# Last-step methods whose extra arguments are routed (the predict-time entry points, plus
# `learn_one` for an unsupervised final transformer that learns during `predict`/`transform`).
_LAST_STEP_METHODS = ("learn_one", "predict_one", "predict_proba_one", "score_one", "transform_one")


@contextlib.contextmanager
def learn_during_predict() -> Iterator[None]:
    """A context manager for fitting unsupervised steps during prediction.

    Usually, unsupervised parts of a pipeline are updated during `learn_one`. However, in the case
    of online learning, it is possible to update them before, during the prediction step. This
    context manager allows you to do so.

    This usually brings a slight performance improvement. But it is not done by default because it
    is not intuitive and is more difficult to test. It also means that you have to call
    `predict_one` before `learn_one` in order for the whole pipeline to be updated.

    Examples
    --------

    Let's first see what methods are called if we just call `predict_one`.

    >>> import io
    >>> import logging
    >>> import pandas as pd
    >>> from river import compose
    >>> from river import datasets
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import utils

    >>> model = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     linear_model.LinearRegression()
    ... )

    >>> class_condition = lambda x: x.__class__.__name__ in ('StandardScaler', 'LinearRegression')

    >>> logger = logging.getLogger()
    >>> logger.setLevel(logging.DEBUG)

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition):
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_one(x)

    >>> print(logs.getvalue())
    StandardScaler.transform_one
    LinearRegression.predict_one

    Now let's use the context manager and see what methods get called.

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition), compose.learn_during_predict():
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_one(x)

    >>> print(logs.getvalue())
    StandardScaler.learn_one
    StandardScaler.transform_one
    LinearRegression.predict_one

    We can see that the scaler did not get updated before transforming the data.

    This also works when working with mini-batches.

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition):
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_many(pd.DataFrame([x]))
    >>> print(logs.getvalue())
    StandardScaler.transform_many
    LinearRegression.predict_many

    >>> logs = io.StringIO()
    >>> sh = logging.StreamHandler(logs)
    >>> sh.setLevel(logging.DEBUG)
    >>> logger.addHandler(sh)

    >>> with utils.log_method_calls(class_condition), compose.learn_during_predict():
    ...     for x, y in datasets.TrumpApproval().take(1):
    ...         _ = model.predict_many(pd.DataFrame([x]))
    >>> print(logs.getvalue())
    StandardScaler.learn_many
    StandardScaler.transform_many
    LinearRegression.predict_many

    """
    Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT = True
    try:
        yield
    finally:
        Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT = False


class Pipeline(base.Estimator):
    """A pipeline of estimators.

    Pipelines allow you to chain different steps into a sequence. Typically, when doing supervised
    learning, a pipeline contains one or more transformation steps, whilst it's a regressor or
    a classifier. It is highly recommended to use pipelines with River. Indeed, in an online
    learning setting, it is very practical to have a model defined as a single object. Take a look
    at the [user guide](/recipes/pipelines) for further information and
    practical examples.

    One special thing to take notice to is the way transformers are handled. It is usual to predict
    something for a sample and wait for the ground truth to arrive. In such a scenario, the
    features are seen before the ground truth arrives. Therefore, the unsupervised parts of the
    pipeline are updated when `predict_one` and `predict_proba_one` are called. Usually the
    unsupervised parts of the pipeline are all the steps that precede the final step, which is a
    supervised model. However, some transformers are supervised and are therefore also updated
    during calls to `learn_one`.

    Parameters
    ----------
    steps
        Ideally, a list of (name, estimator) tuples. A name is automatically inferred if none is
        provided.

    Examples
    --------

    The recommended way to declare a pipeline is to use the `|` operator. The latter allows you
    to chain estimators in a very terse manner:

    >>> from river import linear_model
    >>> from river import preprocessing

    >>> scaler = preprocessing.StandardScaler()
    >>> log_reg = linear_model.LinearRegression()
    >>> model = scaler | log_reg

    This results in a pipeline that stores each step inside a dictionary.

    >>> model
    Pipeline (
      StandardScaler (
        with_std=True
        window_size=None
      ),
      LinearRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=Squared ()
        l2=0.
        l1=0.
        intercept_init=0.
        intercept_lr=Constant (
          learning_rate=0.01
        )
        clip_gradient=1e+12
        initializer=Zeros ()
      )
    )

    You can access parts of a pipeline in the same manner as a dictionary:

    >>> model['LinearRegression']
    LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
        )
      )
      loss=Squared ()
      l2=0.
      l1=0.
      intercept_init=0.
      intercept_lr=Constant (
        learning_rate=0.01
      )
      clip_gradient=1e+12
      initializer=Zeros ()
    )

    Note that you can also declare a pipeline by using the `compose.Pipeline` constructor
    method, which is slightly more verbose:

    >>> from river import compose

    >>> model = compose.Pipeline(scaler, log_reg)

    By using a `compose.TransformerUnion`, you can define complex pipelines that apply
    different steps to different parts of the data. For instance, we can extract word counts
    from text data, and extract polynomial features from numeric data.

    >>> from river import feature_extraction as fx

    >>> tfidf = fx.TFIDF('text')
    >>> counts = fx.BagOfWords('text')
    >>> text_part = compose.Select('text') | (tfidf + counts)

    >>> num_part = compose.Select('a', 'b') | fx.PolynomialExtender()

    >>> model = text_part + num_part
    >>> model |= preprocessing.StandardScaler()
    >>> model |= linear_model.LinearRegression()

    The following shows an example of using `debug_one` to visualize how the information
    flows and changes throughout the pipeline.

    >>> from river import compose
    >>> from river import naive_bayes

    >>> dataset = [
    ...     ('A positive comment', True),
    ...     ('A negative comment', False),
    ...     ('A happy comment', True),
    ...     ('A lovely comment', True),
    ...     ('A harsh comment', False)
    ... ]

    >>> tfidf = fx.TFIDF() | compose.Prefixer('tfidf_')
    >>> counts = fx.BagOfWords() | compose.Prefixer('count_')
    >>> mnb = naive_bayes.MultinomialNB()
    >>> model = (tfidf + counts) | mnb

    >>> for x, y in dataset:
    ...     model.learn_one(x, y)

    >>> x = dataset[0][0]
    >>> report = model.debug_one(dataset[0][0])
    >>> print(report)
    0. Input
    --------
    A positive comment
    1. Transformer union
    --------------------
        1.0 TFIDF | Prefixer
        --------------------
        tfidf_comment: 0.43017 (float)
        tfidf_positive: 0.90275 (float)
        1.1 BagOfWords | Prefixer
        -------------------------
        count_comment: 1 (int)
        count_positive: 1 (int)
    count_comment: 1 (int)
    count_positive: 1 (int)
    tfidf_comment: 0.43017 (float)
    tfidf_positive: 0.90275 (float)
    2. MultinomialNB
    ----------------
    False: 0.19221
    True: 0.80779

    """

    _LEARN_UNSUPERVISED_DURING_PREDICT = False

    def __init__(self, *steps: object) -> None:
        self._steps: collections.OrderedDict[str, base.Estimator | None] = collections.OrderedDict()
        self._plan: tuple[_OnePlan, ...] | None = None
        self._inference_transformers: tuple[_TransformOne, ...] = ()
        self._many_plan: tuple[_ManyPlan, ...] = ()
        self._batch_inference_transformers: tuple[_TransformMany | None, ...] = ()
        self._last_step_cached: base.Estimator | None = None
        self._predict_one_func: _PredictOneFunc | None = None
        self._predict_proba_one_func: _PredictProbaOneFunc | None = None
        self._score_one_func: _ScoreOneFunc | None = None
        self._predict_many_func: _PredictManyFunc | None = None
        self._predict_proba_many_func: _PredictProbaManyFunc | None = None
        self._last_step_params: dict[str, frozenset[str] | None] = {}
        for step in steps:
            self._add_step(step, at_start=False)
        self._validate()
        if self.steps and all(step is not None for step in self.steps.values()):
            self._build_plan()

    @property
    def steps(self) -> Mapping[str, base.Estimator | None]:
        return types.MappingProxyType(self._steps)

    def _build_plan(self) -> tuple[_OnePlan, ...]:
        """Build a cached execution plan describing each step's role.

        Each item is a tuple ``(kind, step, *info)`` where ``kind`` is one of
        ``"anomaly"``, ``"transformer"``, ``"union"``, ``"final"``. This avoids
        repeated `isinstance` checks and ``_supervised`` property lookups on the
        per-event hot path.

        The plan also records, per step, which extra keyword arguments its `learn_one`
        accepts (see `_method_params`). This is the routing "blueprint": signature
        introspection happens here, once per plan, so that `learn_one` (and the predict-time
        methods) can forward arguments such as a `utils.TimeRolling` timestamp `t` to the steps
        that declare them without any per-event inspection.
        """
        plan: list[_OnePlan] = []
        many_plan: list[_ManyPlan] = []
        for step in self.steps.values():
            if step is None:
                raise TypeError("Pipeline placeholders cannot be executed")
            if isinstance(step, base.AnomalyFilter):
                plan.append(
                    (
                        "anomaly",
                        step,
                        step._supervised,
                        step.learn_one,
                        _method_params(step, "learn_one"),
                    )
                )
                many_plan.append(("anomaly", step))
            elif isinstance(step, base.Transformer):
                before = getattr(step, "_learn_one_before", None)
                after = getattr(step, "_learn_one_after", None)
                phased = before is not None or after is not None
                learn = getattr(step, "learn_one", None)
                if not phased and learn is not None:
                    if step._supervised:
                        after = learn
                    else:
                        before = learn
                plan.append(
                    (
                        "transformer",
                        step,
                        step.transform_one,
                        before,
                        after,
                        _method_params(step, "learn_one"),
                        phased,
                    )
                )

                many_before = getattr(step, "_learn_many_before", None)
                many_after = getattr(step, "_learn_many_after", None)
                many_phased = many_before is not None or many_after is not None
                learn_many = getattr(step, "learn_many", None)
                if not many_phased and learn_many is not None:
                    if step._supervised:
                        many_after = learn_many
                    else:
                        many_before = learn_many
                many_plan.append(
                    (
                        "transformer",
                        step,
                        getattr(step, "transform_many", None),
                        many_before,
                        many_after,
                        _method_params(step, "learn_many"),
                        many_phased,
                    )
                )
            else:
                plan.append(
                    (
                        "final",
                        step,
                        step._supervised,
                        getattr(step, "learn_one", None),
                        _method_params(step, "learn_one"),
                    )
                )
                many_plan.append(
                    (
                        "final",
                        step,
                        step._supervised,
                        getattr(step, "learn_many", None),
                        _method_params(step, "learn_many"),
                    )
                )
        self._plan = tuple(plan)
        self._many_plan = tuple(many_plan)
        # Precomputed fast-path inference list: intermediate non-anomaly transformers.
        # Used by `_transform_one` to avoid per-call slicing and anomaly checks.
        self._inference_transformers = tuple(
            item[2] for item in plan[:-1] if item[0] == "transformer"
        )
        self._batch_inference_transformers = tuple(
            item[2] for item in many_plan[:-1] if item[0] == "transformer"
        )
        last = plan[-1][1] if plan else None
        self._last_step_cached = last
        self._predict_one_func = getattr(last, "predict_one", None)
        self._predict_proba_one_func = getattr(last, "predict_proba_one", None)
        self._score_one_func = getattr(last, "score_one", None)
        self._predict_many_func = getattr(last, "predict_many", None)
        self._predict_proba_many_func = getattr(last, "predict_proba_many", None)
        # Accepted extra arguments for the last step's prediction-time methods, so the predict
        # entry points can route the same way `learn_one` does (e.g. drop a learn-only `t`).
        self._last_step_params = (
            {m: _method_params(last, m) for m in _LAST_STEP_METHODS} if last is not None else {}
        )
        return self._plan

    def __getitem__(self, key: str | int) -> Any:
        """Just for convenience."""
        if isinstance(key, int):
            return self.steps[list(self.steps.keys())[key]]
        return self.steps[key]

    def __len__(self) -> int:
        """Just for convenience."""
        return len(self.steps)

    def __or__(self, other: object) -> Pipeline:
        """Insert a step at the end of the pipeline."""
        if isinstance(other, Pipeline):
            return self.__class__(*self.steps.items(), *other.steps.items())
        return self.__class__(*self.steps.items(), other)

    def __ror__(self, other: object) -> Pipeline:
        """Insert a step at the start of the pipeline."""
        if isinstance(other, Pipeline):
            return self.__class__(*other.steps.items(), *self.steps.items())
        return self.__class__(other, *self.steps.items())

    def __add__(self, other: object) -> union.TransformerUnion:
        """Merge with another Pipeline or TransformerUnion into a TransformerUnion."""
        if isinstance(other, union.TransformerUnion):
            return other.__add__(self)
        return union.TransformerUnion(self, other)

    def __mul__(
        self,
        other: base.Transformer
        | Pipeline
        | base.typing.FeatureName
        | list[base.typing.FeatureName],
    ) -> grouper.Grouper | product.TransformerProduct:
        if isinstance(other, base.Transformer) or isinstance(other, Pipeline):
            return product.TransformerProduct(self, other)

        return grouper.Grouper(transformer=self, by=other)

    def __rmul__(
        self,
        other: base.Transformer
        | Pipeline
        | base.typing.FeatureName
        | list[base.typing.FeatureName],
    ) -> grouper.Grouper | product.TransformerProduct:
        return self * other

    def __str__(self) -> str:
        return " | ".join(map(str, self.steps.values()))

    def __repr__(self) -> str:
        return (
            "Pipeline (\n\t"
            + "\t".join(",\n".join(map(repr, self.steps.values())).splitlines(True))
            + "\n)"
        ).expandtabs(2)

    def _get_params(self) -> dict[str, Any]:
        return {
            name: step._get_params() if step is not None else None
            for name, step in self.steps.items()
        }

    def clone(
        self, new_params: dict[str, Any] | None = None, include_attributes: bool = False
    ) -> Pipeline:
        if new_params is None:
            new_params = {}

        steps: list[tuple[str, base.Estimator | None]] = []
        for name, step in self.steps.items():
            replacement = new_params.get(name)
            if isinstance(replacement, base.Estimator):
                steps.append((name, replacement))
            elif step is None:
                steps.append((name, None))
            else:
                steps.append(
                    (
                        name,
                        step.clone(replacement or {}, include_attributes=include_attributes),
                    )
                )
        return self.__class__(*steps)

    def mutate(self, new_params: dict[str, Any]) -> None:
        for step_name, step_params in new_params.items():
            step = self[step_name]
            if step is None:
                raise TypeError("Pipeline placeholders cannot be mutated")
            step.mutate(step_params)

    @property
    def _supervised(self) -> bool:
        return any(step._supervised for step in self.steps.values() if step is not None)

    @property
    def _can_mini_batch(self) -> bool:
        return all(step._can_mini_batch for step in self.steps.values() if step is not None)

    @property
    def _last_step(self) -> base.Estimator:
        step = list(self.steps.values())[-1]
        if step is None:
            raise TypeError("Pipeline placeholders cannot be executed")
        return step

    @property
    def _multiclass(self) -> bool:
        return bool(getattr(self._last_step, "_multiclass", False))

    def _add_step(self, obj: object, at_start: bool) -> None:
        """Add a step to either end of the pipeline.

        This method takes care of sanitizing the input. For instance, if a function is passed,
        then it will be wrapped with a `compose.FuncTransformer`.

        """

        name = None
        if isinstance(obj, tuple):
            name, obj = typing.cast(tuple[str, object], obj)

        def _coerce_to_estimator(obj: object) -> base.Estimator | None:
            if isinstance(obj, types.FunctionType) or isinstance(obj, types.LambdaType):
                return func.FuncTransformer(obj)
            if isinstance(obj, list):
                return union.TransformerUnion(*[_coerce_to_estimator(part) for part in obj])
            if isinstance(obj, type):
                obj = obj()
            if obj is None or isinstance(obj, base.Estimator):
                return obj
            raise TypeError(f"{obj!r} is not an estimator")

        estimator = _coerce_to_estimator(obj)

        def infer_name(estimator: base.Estimator | None) -> str:
            if isinstance(estimator, func.FuncTransformer):
                return estimator.func.__name__
            return estimator.__class__.__name__

        # Infer a name if none is given
        if name is None:
            name = infer_name(estimator)

        # Make sure the name doesn't already exist
        if name in self.steps:
            counter = 1
            while f"{name}{counter}" in self.steps:
                counter += 1
            name = f"{name}{counter}"

        # Store the step
        self._steps[name] = estimator

        # Move the step to the start of the pipeline if so instructed
        if at_start:
            self._steps.move_to_end(name, last=False)

        # Invalidate the cached plan
        self._plan = None

    def _validate(self) -> None:
        if len(self.steps) < 2:
            return
        for name, step in itertools.islice(self.steps.items(), len(self.steps) - 1):
            if step is None:
                continue
            if not isinstance(step, (base.Transformer, base.AnomalyFilter)):
                raise TypeError(f"Pipeline step {name!r} does not transform observations")

    # Single instance methods

    def learn_one(
        self,
        x: dict[Any, Any],
        y: object = None,
        **params: object,
    ) -> None:
        """Fit to a single instance.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A target value.

        """

        plan = self._plan if self._plan is not None else self._build_plan()
        learn_during_predict = self._LEARN_UNSUPERVISED_DURING_PREDICT

        if not params:
            for item in plan:
                if item[0] == "transformer":
                    transformer = item
                    x_pre = x
                    before = transformer[3]
                    if not learn_during_predict and before is not None:
                        if transformer[6]:
                            before(x, params)
                        else:
                            before(x)
                    x = transformer[2](x)
                    after = transformer[4]
                    if after is not None:
                        if transformer[6]:
                            after(x_pre, y, params)
                        else:
                            after(x_pre, y)

                elif item[0] == "final":
                    final = item
                    learn = final[3]
                    if learn is None:
                        continue
                    if final[2]:
                        learn(x=x, y=y)
                    else:
                        learn(x=x)

                else:
                    anomaly = item
                    anomaly_step = anomaly[1]
                    if anomaly[2]:
                        anomaly_step.learn_one(x, y)
                        score = anomaly_step.score_one(x, y)
                    else:
                        anomaly_step.learn_one(x)
                        score = anomaly_step.score_one(x)
                    if anomaly_step.classify(score):
                        break
            return

        for item in plan:
            if item[0] == "transformer":
                transformer = item
                x_pre = x
                before = transformer[3]
                if not learn_during_predict and before is not None:
                    if transformer[6]:
                        before(x, params)
                    else:
                        before(x, **_route(transformer[5], params))
                x = transformer[2](x)
                after = transformer[4]
                if after is not None:
                    if transformer[6]:
                        after(x_pre, y, params)
                    else:
                        after(x=x_pre, y=y, **_route(transformer[5], params))

            elif item[0] == "final":
                final = item
                learn = final[3]
                if learn is None:
                    continue
                if final[2]:
                    learn(x=x, y=y, **_route(final[4], params))
                else:
                    learn(x=x, **_route(final[4], params))

            else:  # "anomaly"
                anomaly = item
                anomaly_step = anomaly[1]
                # There might be an anomaly filter in the pipeline. Its purpose is to prevent
                # anomalous data from being learned by the subsequent parts of the pipeline.
                if anomaly[2]:
                    anomaly_step.learn_one(x, y, **_route(anomaly[4], params))
                    score = anomaly_step.score_one(x, y)
                else:
                    anomaly_step.learn_one(x, **_route(anomaly[4], params))
                    score = anomaly_step.score_one(x)
                if anomaly_step.classify(score):
                    break

    def _transform_one(
        self,
        x: dict[base.typing.FeatureName, Any],
        params: dict[str, object] | None = None,
    ) -> tuple[dict[base.typing.FeatureName, Any], base.Estimator]:
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        `params` are the extra keyword arguments passed to the calling predict-time method; they
        are only consumed on the learn-during-predict slow path, where they are routed to each
        unsupervised step exactly as in `learn_one`.

        """

        plan = self._plan if self._plan is not None else self._build_plan()

        # Fast path: typical inference, no learn-during-predict.
        if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
            for transform in self._inference_transformers:
                x = transform(x)
            if self._last_step_cached is None:
                raise TypeError("Pipeline placeholders cannot be executed")
            return x, self._last_step_cached

        # Slow path: learn unsupervised steps before transforming.
        extra = params or {}
        for item in plan[:-1]:
            if item[0] != "transformer":
                continue
            plan_item = item
            step = plan_item[1]
            before = plan_item[3]
            if before is not None:
                if plan_item[6]:
                    before(x, extra)
                else:
                    before(x, **_route(plan_item[5], extra))
            x = step.transform_one(x)

        if self._last_step_cached is None:
            raise TypeError("Pipeline placeholders cannot be executed")
        return x, self._last_step_cached

    def transform_one(
        self,
        x: dict[Any, Any],
        **params: object,
    ) -> dict[base.typing.FeatureName, Any]:
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        x, last_step = self._transform_one(x, params)
        if isinstance(last_step, base.Transformer):
            if not last_step._supervised and self._LEARN_UNSUPERVISED_DURING_PREDICT:
                learn = getattr(last_step, "learn_one", None)
                if learn is not None:
                    learn(x, **_route(self._last_step_params["learn_one"], params))
            return last_step.transform_one(
                x, **_route(self._last_step_params["transform_one"], params)
            )
        return x

    def predict_one(
        self,
        x: dict[Any, Any],
        **params: object,
    ) -> Any:
        """Call `transform_one` on the first steps and `predict_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, _ = self._transform_one(x, params)
        if self._predict_one_func is None:
            raise AttributeError("The pipeline has no final predictor")
        if not params:
            return self._predict_one_func(x)
        return self._predict_one_func(x, **_route(self._last_step_params["predict_one"], params))

    def predict_proba_one(
        self,
        x: dict[Any, Any],
        **params: object,
    ) -> dict[base.typing.ClfTarget, float]:
        """Call `transform_one` on the first steps and `predict_proba_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, _ = self._transform_one(x, params)
        if self._predict_proba_one_func is None:
            raise AttributeError("The pipeline has no final probabilistic predictor")
        result: dict[base.typing.ClfTarget, float]
        if not params:
            result = self._predict_proba_one_func(x)
        else:
            result = self._predict_proba_one_func(
                x, **_route(self._last_step_params["predict_proba_one"], params)
            )
        return result

    def score_one(
        self,
        x: dict[Any, Any],
        **params: object,
    ) -> float:
        """Call `transform_one` on the first steps and `score_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, _ = self._transform_one(x, params)
        if self._score_one_func is None:
            raise AttributeError("The pipeline has no final scorer")
        if not params:
            return self._score_one_func(x)
        return self._score_one_func(x, **_route(self._last_step_params["score_one"], params))

    def forecast(
        self,
        horizon: int,
        xs: list[dict[base.typing.FeatureName, Any]] | None = None,
    ) -> list[float]:
        """Return a forecast.

        Only works if each estimator has a `transform_one` method and the final estimator has a
        `forecast` method. This is the case of time series models from the `time_series` module.

        Parameters
        ----------
        horizon
            The forecast horizon.
        xs
            A list of features for each step in the horizon.

        """
        if xs is not None:
            xs = [self._transform_one(x)[0] for x in xs]
        forecast = getattr(self._last_step, "forecast")
        return typing.cast(list[float], forecast(horizon=horizon, xs=xs))

    def debug_one(
        self,
        x: dict[base.typing.FeatureName, Any],
        show_types: bool = True,
        n_decimals: int = 5,
    ) -> str:
        """Displays the state of a set of features as it goes through the pipeline.

        Parameters
        ----------
        x
            A set of features.
        show_types
            Whether or not to display the type of feature along with it's value.
        n_decimals
            Number of decimals to display for each floating point value.

        """

        tab = " " * 4

        # We'll redirect all the print statement to a buffer, we'll return the content of the
        # buffer at the end
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        def format_value(x: object) -> object:
            if isinstance(x, float):
                return "{:,.{prec}f}".format(x, prec=n_decimals)
            return x

        def print_dict(
            x: Any, show_types: bool, indent: bool = False, space_after: bool = True
        ) -> None:
            # Some transformers accept strings as input instead of dicts
            if isinstance(x, str):
                _print(x)
            else:
                for k, v in sorted(x.items()):
                    type_str = f" ({type(v).__name__})" if show_types else ""
                    _print((tab if indent else "") + f"{k}: {format_value(v)}" + type_str)
            if space_after:
                _print()

        def print_title(title: str, indent: bool = False) -> None:
            _print((tab if indent else "") + title)
            _print((tab if indent else "") + "-" * len(title))

        # Print the initial state of the features
        print_title("0. Input")
        print_dict(x, show_types=show_types)

        # Print the state of x at each step
        steps = iter(self.steps.values())
        for i, t in enumerate(itertools.islice(steps, len(self) - 1)):
            if isinstance(t, union.TransformerUnion):
                print_title(f"{i + 1}. Transformer union")
                for j, (name, sub_t) in enumerate(t.transformers.items()):
                    if isinstance(sub_t, Pipeline):
                        name = str(sub_t)
                    print_title(f"{i + 1}.{j} {name}", indent=True)
                    print_dict(
                        typing.cast(base.Transformer, sub_t).transform_one(x),
                        show_types=show_types,
                        indent=True,
                    )
                x = t.transform_one(x)
                print_dict(x, show_types=show_types)

            else:
                print_title(f"{i + 1}. {t}")
                x = typing.cast(base.typing.Transformer, t).transform_one(x)
                print_dict(x, show_types=show_types)

        # Print the predicted output from the final estimator
        final = next(steps)
        if not isinstance(final, base.Transformer):
            print_title(f"{len(self)}. {final}")

            # If the last estimator has a debug_one method then call it
            debug = getattr(final, "debug_one", None)
            if debug is not None:
                _print(debug(x))
                _print()

            if isinstance(final, base.Classifier):
                print_dict(final.predict_proba_one(x), show_types=False, space_after=False)
            elif isinstance(final, base.AnomalyDetector):
                _print(f"Score: {format_value(final.score_one(x))}")
            else:
                prediction = typing.cast(base.typing.Predictor, final).predict_one(x)
                _print(f"Prediction: {format_value(prediction)}")

        return buffer.getvalue().rstrip()

    # Mini-batch methods

    def learn_many(
        self,
        X: IntoDataFrame,
        y: IntoSeries | None = None,
        **params: object,
    ) -> None:
        """Fit to a mini-batch.

        Parameters
        ----------
        X
            A dataframe of features. Columns can be added and/or removed between successive calls.
        y
            A series of target values.

        """

        if self._plan is None:
            self._build_plan()

        for item in self._many_plan:
            if item[0] == "transformer":
                transformer = item
                step = transformer[1]
                X_pre = X
                before = transformer[3]
                if not self._LEARN_UNSUPERVISED_DURING_PREDICT and before is not None:
                    if transformer[6]:
                        before(X, params)
                    else:
                        before(X, **_route(transformer[5], params))
                transform = transformer[2]
                if transform is None:
                    raise AttributeError(f"{step} does not implement transform_many")
                X = transform(X)
                after = transformer[4]
                if after is not None:
                    if transformer[6]:
                        after(X_pre, y, params)
                    else:
                        after(X_pre, y, **_route(transformer[5], params))
            elif item[0] == "final":
                final = item
                learn = final[3]
                if learn is None:
                    raise AttributeError(f"{final[1]} does not implement learn_many")
                if final[2]:
                    learn(X, y, **_route(final[4], params))
                else:
                    learn(X, **_route(final[4], params))
            else:
                raise AttributeError(f"{item[1]} does not implement mini-batch learning")

    def _transform_many(self, X: IntoDataFrame) -> tuple[IntoDataFrame, base.Estimator]:
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        if self._plan is None:
            self._build_plan()

        if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
            for transform in self._batch_inference_transformers:
                if transform is None:
                    raise AttributeError("A pipeline step does not implement transform_many")
                X = transform(X)
            if self._last_step_cached is None:
                raise TypeError("Pipeline placeholders cannot be executed")
            return X, self._last_step_cached

        for item in self._many_plan[:-1]:
            if item[0] != "transformer":
                continue
            plan_item = item
            step = plan_item[1]
            before = plan_item[3]
            if before is not None:
                if plan_item[6]:
                    before(X, {})
                else:
                    before(X)
            transform = plan_item[2]
            if transform is None:
                raise AttributeError(f"{step} does not implement transform_many")
            X = transform(X)

        if self._last_step_cached is None:
            raise TypeError("Pipeline placeholders cannot be executed")
        return X, self._last_step_cached

    def transform_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        X, last_step = self._transform_many(X=X)
        if isinstance(last_step, base.typing.MiniBatchTransformer):
            if not last_step._supervised and self._LEARN_UNSUPERVISED_DURING_PREDICT:
                learn = getattr(last_step, "learn_many", None)
                if learn is not None:
                    learn(X)
            return last_step.transform_many(X)
        return X

    def predict_many(self, X: IntoDataFrame) -> IntoSeries:
        """Call transform_many, and then predict_many on the final step."""
        X, _ = self._transform_many(X=X)
        if self._predict_many_func is None:
            raise AttributeError("The pipeline has no final mini-batch predictor")
        return self._predict_many_func(X=X)

    def predict_proba_many(self, X: IntoDataFrame) -> IntoDataFrame:
        """Call transform_many, and then predict_proba_many on the final step."""
        X, _ = self._transform_many(X=X)
        if self._predict_proba_many_func is None:
            raise AttributeError("The pipeline has no final mini-batch probabilistic predictor")
        return self._predict_proba_many_func(X=X)

    def _unit_test_skips(self) -> set[str]:
        step = self[-1]
        if step is None:
            return set()
        return typing.cast(set[str], step._unit_test_skips())
