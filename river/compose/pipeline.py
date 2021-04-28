import collections
import functools
import io
import itertools
import types
import typing
from xml.etree import ElementTree as ET

import pandas as pd

from .. import base, utils
from . import func, union

__all__ = ["Pipeline"]


class Pipeline(base.Estimator):
    """A pipeline of estimators.

    Pipelines allow you to chain different steps into a sequence. Typically, when doing supervised
    learning, a pipeline contains one ore more transformation steps, whilst it's is a regressor or
    a classifier. It is highly recommended to use pipelines with `river`. Indeed, in an online
    learning setting, it is very practical to have a model defined as a single object. Take a look
    at the [user guide](../../user-guide/pipelines.md) for further information and
    practical examples.

    One special thing to take notice to is the way transformers are handled. In a typical scenario,
    it is usual to predict something for a sample and wait for the ground truth to arrive. In such
    a case, the features are seen before the ground truth arrives. Therefore, the unsupervised
    parts of the pipeline are updated when `predict_one` and `predict_proba_one` are called.
    Usually the unsupervised parts of the pipeline are all the steps that precede the final step,
    which is a supervised model. However, some transformers are supervised and are therefore
    obtained during calls to `learn_one`.

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
      StandardScaler (),
      LinearRegression (
        optimizer=SGD (
          lr=Constant (
            learning_rate=0.01
          )
        )
        loss=Squared ()
        l2=0.
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

    >>> tfidf = fx.TFIDF() | compose.Renamer(prefix='tfidf_')
    >>> counts = fx.BagOfWords() | compose.Renamer(prefix='count_')
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
    <BLANKLINE>
    1. Transformer union
    --------------------
        1.0 TFIDF | Renamer
        -------------------
        tfidf_comment: 0.47606 (float)
        tfidf_positive: 0.87942 (float)
    <BLANKLINE>
        1.1 BagOfWords | Renamer
        ------------------------
        count_comment: 1 (int)
        count_positive: 1 (int)
    <BLANKLINE>
    count_comment: 1 (int)
    count_positive: 1 (int)
    tfidf_comment: 0.50854 (float)
    tfidf_positive: 0.86104 (float)
    <BLANKLINE>
    2. MultinomialNB
    ----------------
    False: 0.19313
    True: 0.80687

    """

    def __init__(self, *steps):
        self.steps = collections.OrderedDict()
        for step in steps:
            self |= step

    def __getitem__(self, key):
        """Just for convenience."""
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

    def __str__(self):
        return " | ".join(map(str, self.steps.values()))

    def __repr__(self):
        return (
            "Pipeline (\n\t"
            + "\t".join(",\n".join(map(repr, self.steps.values())).splitlines(True))
            + "\n)"
        ).expandtabs(2)

    def _repr_html_(self):

        from river.compose import viz

        html = ET.Element("html")
        body = ET.Element("body")
        html.append(body)

        pipeline_div = viz.pipeline_to_html(self)
        body.append(pipeline_div)

        return f"<html>{ET.tostring(body).decode()}<style>{viz.CSS}</style></html>"

    def _get_params(self):
        return {name: step._get_params() for name, step in self.steps.items()}

    def _set_params(self, new_params: dict = None):

        if new_params is None:
            new_params = {}

        return Pipeline(
            *[
                (name, new_params[name])
                if isinstance(new_params.get(name), base.Estimator)
                else (name, step._set_params(new_params.get(name, {})))
                for name, step in self.steps.items()
            ]
        )

    @property
    def _supervised(self):
        return any(step._supervised for step in self.steps.values())

    @property
    def _multiclass(self):
        return list(self.steps.values())[-1]._multiclass

    def _add_step(self, estimator, at_start: bool):
        """Add a step to either end of the pipeline.

        This method takes care of sanitizing the input. For instance, if a function is passed,
        then it will be wrapped with a `compose.FuncTransformer`.

        """

        name = None
        if isinstance(estimator, tuple):
            name, estimator = estimator

        # If the step is a function then wrap it in a FuncTransformer
        if isinstance(estimator, (types.FunctionType, types.LambdaType)):
            estimator = func.FuncTransformer(estimator)

        def infer_name(estimator):
            if isinstance(estimator, func.FuncTransformer):
                return infer_name(estimator.func)
            elif isinstance(estimator, (types.FunctionType, types.LambdaType)):
                return estimator.__name__
            elif hasattr(estimator, "__class__"):
                return estimator.__class__.__name__
            return str(estimator)

        # Infer a name if none is given
        if name is None:
            name = infer_name(estimator)

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

    def learn_one(self, x: dict, y=None, learn_unsupervised=False, **params):
        """Fit to a single instance.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A target value.
        learn_unsupervised
            Whether the unsupervised parts of the pipeline should be updated or not. See the
            docstring of this class for more information.

        """

        steps = iter(self.steps.values())

        # Loop over the first n - 1 steps, which should all be transformers
        for t in itertools.islice(steps, len(self.steps) - 1):
            x_pre = x
            x = t.transform_one(x=x)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if sub_t._supervised:
                        sub_t.learn_one(x=x_pre, y=y)
                    elif learn_unsupervised:
                        sub_t.learn_one(x=x_pre)

            elif t._supervised:
                t.learn_one(x=x_pre, y=y)

            elif learn_unsupervised:
                t.learn_one(x=x_pre)

        # At this point steps contains a single step, which is therefore the final step of the
        # pipeline
        final = next(steps)
        if final._supervised:
            final.learn_one(x=x, y=y, **params)
        elif learn_unsupervised:
            final.learn_one(x=x, **params)

        return self

    def _transform_one(self, x: dict, learn_unsupervised=True):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self.steps) - 1):

            # The unsupervised transformers are updated during transform. We do this because
            # typically transform_one is called before learn_one, and therefore we might as well use
            # the available information as soon as possible. Note that way of proceeding is very
            # specific to online machine learning.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if not sub_t._supervised and learn_unsupervised:
                        sub_t.learn_one(x=x)

            elif not t._supervised and learn_unsupervised:
                t.learn_one(x=x)

            x = t.transform_one(x=x)

        final_step = next(steps)
        if not final_step._supervised and learn_unsupervised:
            final_step.learn_one(x)

        return x, final_step

    def transform_one(self, x: dict):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        x, final_step = self._transform_one(x=x)
        if isinstance(final_step, base.Transformer):
            return final_step.transform_one(x=x)
        return x

    def predict_one(self, x: dict, learn_unsupervised=True):
        """Call `transform_one` on the first steps and `predict_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.
        learn_unsupervised
            Whether the unsupervised parts of the pipeline should be updated or not. See the
            docstring of this class for more information.

        """
        x, final_step = self._transform_one(x=x, learn_unsupervised=learn_unsupervised)
        return final_step.predict_one(x=x)

    def predict_proba_one(self, x: dict, learn_unsupervised=True):
        """Call `transform_one` on the first steps and `predict_proba_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.
        learn_unsupervised
            Whether the unsupervised parts of the pipeline should be updated or not. See the
            docstring of this class for more information.

        """
        x, final_step = self._transform_one(x=x, learn_unsupervised=learn_unsupervised)
        return final_step.predict_proba_one(x=x)

    def score_one(self, x: dict, learn_unsupervised=True):
        """Call `transform_one` on the first steps and `score_one` on the last step.

        Parameters
        ----------
        x
            A dictionary of features.
        learn_unsupervised
            Whether the unsupervised parts of the pipeline should be updated or not. See the
            docstring of this class for more information.

        """
        x, final_step = self._transform_one(x=x, learn_unsupervised=learn_unsupervised)
        return final_step.score_one(x=x)

    def forecast(self, horizon: int, xs: typing.List[dict] = None):
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
        final_step = list(self.steps.values())[-1]
        return final_step.forecast(horizon=horizon, xs=xs)

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
                    _print(
                        (tab if indent else "") + f"{k}: {format_value(v)}" + type_str
                    )
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
        for i, t in enumerate(itertools.islice(steps, len(self.steps) - 1)):

            if isinstance(t, union.TransformerUnion):
                print_title(f"{i+1}. Transformer union")
                for j, (name, sub_t) in enumerate(t.transformers.items()):
                    if isinstance(sub_t, Pipeline):
                        name = str(sub_t)
                    print_title(f"{i+1}.{j} {name}", indent=True)
                    print_dict(
                        sub_t.transform_one(x), show_types=show_types, indent=True
                    )
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
                print_dict(
                    final.predict_proba_one(x), show_types=False, space_after=False
                )
            else:
                _print(f"Prediction: {format_value(final.predict_one(x))}")

        return buffer.getvalue().rstrip()

    # Mini-batch methods

    def learn_many(
        self, X: pd.DataFrame, y: pd.Series = None, learn_unsupervised=False, **params
    ):
        """Fit to a mini-batch.

        Parameters
        ----------
        X
            A dataframe of features. Columns can be added and/or removed between successive calls.
        y
            A series of target values.
        learn_unsupervised
            Whether the unsupervised parts of the pipeline should be updated or not. See the
            docstring of this class for more information.

        """

        steps = iter(self.steps.values())

        # Loop over the first n - 1 steps, which should all be transformers
        for t in itertools.islice(steps, len(self.steps) - 1):
            X_pre = X
            X = t.transform_many(X=X)

            # The supervised transformers have to be updated.
            # Note that this is done after transforming in order to avoid target leakage.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if sub_t._supervised:
                        sub_t.learn_many(X=X_pre, y=y)
                    elif learn_unsupervised:
                        sub_t.learn_many(X=X_pre)

            elif t._supervised:
                t.learn_many(X=X_pre, y=y)

            elif learn_unsupervised:
                t.learn_many(X=X_pre)

        # At this point steps contains a single step, which is therefore the final step of the
        # pipeline
        final = next(steps)
        if final._supervised:
            final.learn_many(X=X, y=y, **params)
        elif learn_unsupervised:
            final.learn_many(X=X, **params)

        return self

    def _transform_many(self, X: pd.DataFrame, learn_unsupervised=True):
        """This methods takes care of applying the first n - 1 steps of the pipeline, which are
        supposedly transformers. It also returns the final step so that other functions can do
        something with it.

        """

        steps = iter(self.steps.values())

        for t in itertools.islice(steps, len(self.steps) - 1):

            # The unsupervised transformers are updated during transform. We do this because
            # typically transform_one is called before learn_one, and therefore we might as well use
            # the available information as soon as possible. Note that way of proceeding is very
            # specific to online machine learning.
            if isinstance(t, union.TransformerUnion):
                for sub_t in t.transformers.values():
                    if not sub_t._supervised and learn_unsupervised:
                        sub_t.learn_many(X=X)

            elif not t._supervised and learn_unsupervised:
                t.learn_many(X=X)

            X = t.transform_many(X=X)

        return X, next(steps)

    def transform_many(self, X: pd.DataFrame):
        """Apply each transformer in the pipeline to some features.

        The final step in the pipeline will be applied if it is a transformer. If not, then it will
        be ignored and the output from the penultimate step will be returned. Note that the steps
        that precede the final step are assumed to all be transformers.

        """
        X, final_step = self._transform_many(X=X)
        if isinstance(final_step, base.Transformer):
            return final_step.transform_many(X=X)
        return X

    def predict_many(self, X: pd.DataFrame, learn_unsupervised=True):
        X, final_step = self._transform_many(X=X, learn_unsupervised=learn_unsupervised)
        return final_step.predict_many(X=X)

    def predict_proba_many(self, X: pd.DataFrame, learn_unsupervised=True):
        X, final_step = self._transform_many(X=X, learn_unsupervised=learn_unsupervised)
        return final_step.predict_proba_many(X=X)
