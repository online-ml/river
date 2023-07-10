from __future__ import annotations

import collections
import contextlib
import functools
import io
import itertools
import types
import typing

import pandas as pd

from river import base, utils

from . import func, union

__all__ = ["Pipeline"]


@contextlib.contextmanager
def learn_during_predict():
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
    learning, a pipeline contains one ore more transformation steps, whilst it's is a regressor or
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
    ...     model = model.learn_one(x, y)

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

    def __init__(self, *steps):
        self.steps = collections.OrderedDict()
        for step in steps:
            self |= step

    def __getitem__(self, key):
        """Just for convenience."""
        if isinstance(key, int):
            return self.steps[list(self.steps.keys())[key]]
        return self.steps[key]

    def __len__(self):
        """Just for convenience."""
        return len(self.steps)

    def __or__(self, other):
        """Insert a step at the end of the pipeline."""
        self._add_step(other, at_start=False)
        return self

    def __ror__(self, other):
        """Insert a step at the start of the pipeline."""
        self._add_step(other, at_start=True)
        return self

    def __add__(self, other):
        """Merge with another Pipeline or TransformerUnion into a TransformerUnion."""
        if isinstance(other, union.TransformerUnion):
            return other.__add__(self)
        return union.TransformerUnion(self, other)

    def __mul__(self, other):
        from river import compose

        if isinstance(other, base.Transformer) or isinstance(other, Pipeline):
            return compose.TransformerProduct(self, other)

        return compose.Grouper(transformer=self, by=other)

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return " | ".join(map(str, self.steps.values()))

    def __repr__(self):
        return (
            "Pipeline (\n\t"
            + "\t".join(",\n".join(map(repr, self.steps.values())).splitlines(True))
            + "\n)"
        ).expandtabs(2)

    def _get_params(self):
        return {name: step._get_params() for name, step in self.steps.items()}

    def clone(self, new_params: dict | None = None, include_attributes=False):
        if new_params is None:
            new_params = {}

        return self.__class__(
            *[
                (name, new_params[name])
                if isinstance(new_params.get(name), base.Estimator)
                else (
                    name,
                    step.clone(new_params.get(name, {}), include_attributes=include_attributes),
                )
                for name, step in self.steps.items()
            ]
        )

    def mutate(self, new_params: dict):
        for step_name, step_params in new_params.items():
            self[step_name].mutate(step_params)

    @property
    def _supervised(self):
        return any(step._supervised for step in self.steps.values())

    @property
    def _last_step(self):
        return list(self.steps.values())[-1]

    @property
    def _multiclass(self):
        return self._last_step._multiclass

    def _add_step(self, obj: typing.Any, at_start: bool):
        """Add a step to either end of the pipeline.

        This method takes care of sanitizing the input. For instance, if a function is passed,
        then it will be wrapped with a `compose.FuncTransformer`.

        """

        name = None
        if isinstance(obj, tuple):
            name, obj = obj

        def _coerce_to_estimator(obj: typing.Any) -> base.Estimator:
            if isinstance(obj, types.FunctionType) or isinstance(obj, types.LambdaType):
                return func.FuncTransformer(obj)
            if isinstance(obj, list):
                return union.TransformerUnion(*[_coerce_to_estimator(part) for part in obj])
            return obj

        estimator = _coerce_to_estimator(obj)

        def infer_name(estimator: base.Estimator | typing.Callable) -> str:
            if isinstance(estimator, func.FuncTransformer):
                return infer_name(estimator.func)
            if isinstance(estimator, types.FunctionType) or isinstance(obj, types.LambdaType):
                return estimator.__name__  # type: ignore
            if hasattr(estimator, "__class__"):
                return estimator.__class__.__name__
            return str(estimator)

        # Infer a name if none is given
        if name is None:
            name = infer_name(estimator)

        # Make sure the name doesn't already exist
        if name in self.steps:
            counter = 1
            while f"{name}{counter}" in self.steps:
                counter += 1
            name = f"{name}{counter}"

        # Instantiate the estimator if it hasn't been done
        if isinstance(estimator, type):
            estimator = estimator()

        # Store the step
        self.steps[name] = estimator

        # Move the step to the start of the pipeline if so instructed
        if at_start:
            self.steps.move_to_end(name, last=False)

    # Single instance methods

    def learn_one(self, x: dict, y=None, **params):
        """Fit to a single instance.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A target value.

        """

        # Loop over the first n - 1 steps, which should all be transformers
        for step in self.steps.values():
            # There might be an anomaly filter in the pipeline. Its purpose is to prevent anomalous
            # data from being learned by the subsequent parts of the pipeline.
            if utils.inspect.ischildobject(obj=step, class_name="AnomalyFilter"):
                if step._supervised:
                    step.learn_one(x, y)
                    score = step.score_one(x, y)
                else:
                    step.learn_one(x)
                    score = step.score_one(x)
                # Skip the next parts of the pipeline if the score is classified as anomalous
                if step.classify(score):
                    break
                continue

            x_pre = x
            if isinstance(step, base.Transformer):
                # In case of _LEARN_UNSUPERVISED_DURING_PREDICT, then the unsupervised transformers
                # are updated before transforming.
                if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                    if isinstance(step, union.TransformerUnion):
                        for sub_step in step.transformers.values():
                            if not sub_step._supervised:
                                sub_step.learn_one(x)
                    elif not step._supervised:
                        step.learn_one(x)
                # Transform the data
                x = step.transform_one(x)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(step, base.Transformer):
                if isinstance(step, union.TransformerUnion):
                    for sub_step in step.transformers.values():
                        if sub_step._supervised:
                            sub_step.learn_one(x=x_pre, y=y)
                # Here the step is a supervised transformer, such as a TargetAgg. It's important
                # to pass the original features to the transformer, not the transformed ones.
                elif step._supervised:
                    step.learn_one(x=x_pre, y=y)  # type: ignore
            # Here the step is not a transformer, and it's supervised, such as a LinearRegression.
            # This is usually the last step of the pipeline.
            elif step._supervised:
                step.learn_one(x=x, y=y)
            # Here the step is not a transformer, and it's unsupervised, such as a KMeans. This
            # is also usually the last step of the pipeline.
            else:
                step.learn_one(x=x)

        return self

    def _transform_one(self, x: dict):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self) - 1):
            # An anomaly filter is a no-op during inference
            if utils.inspect.ischildobject(obj=t, class_name="AnomalyFilter"):
                continue

            # In case of _LEARN_UNSUPERVISED_DURING_PREDICT, then the unsupervised transformers
            # are updated during the inference phase.
            if self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_one(x)
                elif not t._supervised:
                    t.learn_one(x)

            x = t.transform_one(x)

        last_step = next(steps)
        return x, last_step

    def transform_one(self, x: dict, **params):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        x, last_step = self._transform_one(x)
        if isinstance(last_step, base.Transformer):
            if not last_step._supervised and self._LEARN_UNSUPERVISED_DURING_PREDICT:
                last_step.learn_one(x)
            return last_step.transform_one(x, **params)
        return x

    def predict_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `predict_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.predict_one(x, **params)

    def predict_proba_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `predict_proba_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.predict_proba_one(x, **params)

    def score_one(self, x: dict, **params):
        """Call `transform_one` on the first steps and `score_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.

        """
        x, last_step = self._transform_one(x)
        return last_step.score_one(x, **params)

    def forecast(self, horizon: int, xs: list[dict] | None = None):
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
        return self._last_step.forecast(horizon=horizon, xs=xs)

    def debug_one(self, x: dict, show_types=True, n_decimals=5) -> str:
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

        def format_value(x):
            if isinstance(x, float):
                return "{:,.{prec}f}".format(x, prec=n_decimals)
            return x

        def print_dict(x, show_types, indent=False, space_after=True):
            # Some transformers accept strings as input instead of dicts
            if isinstance(x, str):
                _print(x)
            else:
                for k, v in sorted(x.items()):
                    type_str = f" ({type(v).__name__})" if show_types else ""
                    _print((tab if indent else "") + f"{k}: {format_value(v)}" + type_str)
            if space_after:
                _print()

        def print_title(title, indent=False):
            _print((tab if indent else "") + title)
            _print((tab if indent else "") + "-" * len(title))

        # Print the initial state of the features
        print_title("0. Input")
        print_dict(x, show_types=show_types)

        # Print the state of x at each step
        steps = iter(self.steps.values())
        for i, t in enumerate(itertools.islice(steps, len(self) - 1)):
            if isinstance(t, union.TransformerUnion):
                print_title(f"{i+1}. Transformer union")
                for j, (name, sub_t) in enumerate(t.transformers.items()):
                    if isinstance(sub_t, Pipeline):
                        name = str(sub_t)
                    print_title(f"{i+1}.{j} {name}", indent=True)
                    print_dict(sub_t.transform_one(x), show_types=show_types, indent=True)
                x = t.transform_one(x)
                print_dict(x, show_types=show_types)

            else:
                print_title(f"{i+1}. {t}")
                x = t.transform_one(x)
                print_dict(x, show_types=show_types)

        # Print the predicted output from the final estimator
        final = next(steps)
        if not utils.inspect.istransformer(final):
            print_title(f"{len(self)}. {final}")

            # If the last estimator has a debug_one method then call it
            if hasattr(final, "debug_one"):
                _print(final.debug_one(x))

            # Display the prediction
            _print()
            if utils.inspect.isclassifier(final):
                print_dict(final.predict_proba_one(x), show_types=False, space_after=False)
            elif utils.inspect.isanomalydetector(final):
                _print(f"Score: {format_value(final.score_one(x))}")
            else:
                _print(f"Prediction: {format_value(final.predict_one(x))}")

        return buffer.getvalue().rstrip()

    # Mini-batch methods

    def learn_many(self, X: pd.DataFrame, y: pd.Series | None = None, **params):
        """Fit to a mini-batch.

        Parameters
        ----------
        X
            A dataframe of features. Columns can be added and/or removed between successive calls.
        y
            A series of target values.

        """

        steps = iter(self.steps.values())

        # Loop over the first n - 1 steps, which should all be transformers
        for t in itertools.islice(steps, len(self) - 1):
            if not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_many(X)
                elif not t._supervised:
                    t.learn_many(X)

            X_pre = X
            X = t.transform_many(X=X)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if sub_t._supervised:
                        sub_t.learn_many(X=X_pre, y=y)

            elif t._supervised:
                t.learn_many(X=X_pre, y=y)

        last_step = next(steps)
        if last_step._supervised:
            last_step.learn_many(X=X, y=y, **params)
        else:
            last_step.learn_many(X=X, **params)

        return self

    def _transform_many(self, X: pd.DataFrame):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self) - 1):
            if self._LEARN_UNSUPERVISED_DURING_PREDICT:
                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.transformers.values():
                        if not sub_t._supervised:
                            sub_t.learn_many(X=X)

                elif not t._supervised:
                    t.learn_many(X=X)

            X = t.transform_many(X=X)

        last_step = next(steps)
        return X, last_step

    def transform_many(self, X: pd.DataFrame):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        X, last_step = self._transform_many(X=X)
        if isinstance(last_step, base.MiniBatchTransformer):
            if not last_step._supervised and not self._LEARN_UNSUPERVISED_DURING_PREDICT:
                last_step.learn_many(X)
            return last_step.transform_many(X)
        return X

    def predict_many(self, X: pd.DataFrame):
        """Call transform_many, and then predict_many on the final step."""
        X, last_step = self._transform_many(X=X)
        return last_step.predict_many(X=X)

    def predict_proba_many(self, X: pd.DataFrame):
        """Call transform_many, and then predict_proba_many on the final step."""
        X, last_step = self._transform_many(X=X)
        return last_step.predict_proba_many(X=X)

    def _unit_test_skips(self):
        return self[-1]._unit_test_skips()
