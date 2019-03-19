import collections
import itertools

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False
from sklearn.utils import metaestimators

from .. import base

from . import func
from . import union


__all__ = ['Pipeline']


class Pipeline(collections.OrderedDict):
    """A sequence of estimators.

    Parameters:
        steps (list): Ideally a list of (name, estimator) tuples. If an estimator is given without
            a name then a name is automatically inferred from the estimator.

    """

    def __init__(self, steps=None):
        if steps is not None:
            for step in steps:
                self |= step

    def __or__(self, other):
        """Inserts a step at the end of the pipeline."""
        self.add_step(other, at_start=False)
        return self

    def __ror__(self, other):
        """Inserts a step at the start of the pipeline."""
        self.add_step(other, at_start=True)
        return self

    def __str__(self):
        """Return a human friendly representation of the pipeline."""
        return ' | '.join(self.keys())

    @property
    def __class__(self):
        """Returns the class of the final estimator for type checking purposes.

        A Pipeline is semantically equivalent to it's final estimator in terms of usage. This is
        mostly used for deceiving the ``isinstance`` method.

        """
        return self.final_estimator.__class__

    def add_step(self, step, at_start):
        """Adds a step to either end of the pipeline while taking care of the input type."""

        # Infer a name if none is given
        if not isinstance(step, (list, tuple)):
            step = (str(step), step)

        # If a function is given then wrap it in a FuncTransformer
        if callable(step[1]):
            step = (step[1].__class__, func.FuncTransformer(step[1]))

        # Prefer clarity to magic
        if step[0] in self:
            raise KeyError(f'{step[0]} already exists')

        # Store the step
        self[step[0]] = step[1]

        # Move the step to the start of the pipeline if so instructed
        if at_start:
            self.move_to_end(step[0], last=False)

    @property
    def final_estimator(self):
        """The final estimator."""
        return self[next(reversed(self))]

    def fit_one(self, x, y=None):
        """Fits each step with ``x``."""

        for estimator in self.values():

            if isinstance(estimator, base.Transformer):

                # If a Transformer is supervised then it has to transform the output before fitting
                # in order to prevent target leakage
                if estimator.is_supervised:
                    x = estimator.transform_one(x)
                    estimator.fit_one(x, y)
                else:
                    x = estimator.fit_one(x).transform_one(x)
            else:
                estimator.fit_one(x, y)

        return self

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def transform_one(self, x):
        """Transform an input.

        Only works if each estimator has a ``transform_one`` method.

        """
        for estimator in self.values():
            x = estimator.transform_one(x)
        return x

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_one(self, x):
        """Predict output.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a ``predict_one`` method.

        """
        for estimator in itertools.islice(self.values(), len(self) - 1):
            x = estimator.transform_one(x)
        return self.final_estimator.predict_one(x)

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_proba_one(self, x):
        """Predicts probabilities.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a ``predict_proba_one`` method.

        """
        for estimator in itertools.islice(self.values(), len(self) - 1):
            x = estimator.transform_one(x)
        return self.final_estimator.predict_proba_one(x)

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def fit_transform_one(self, x, y=None):
        """Fit and transform.

        Only works if each estimator has a ``fit_transform_one`` method.

        """
        for estimator in self.values():
            x = estimator.fit_transform_one(x, y)
        return x

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def fit_predict_one(self, x, y):
        """Fit and predict.

        Only works if each estimator has a ``fit_transform_one`` method and the final estimator has
        a ``fit_predict_one`` method.

        """
        for estimator in itertools.islice(self.values(), len(self) - 1):
            x = estimator.fit_transform_one(x, y)
        return self.final_estimator.fit_predict_one(x, y)

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def fit_predict_proba_one(self, x, y):
        """Fit and predict probabilities.

        Only works if each estimator has a ``fit_transform_one`` method and the final estimator has
        a ``fit_predict_proba_one`` method.

        """
        for estimator in itertools.islice(self.values(), len(self) - 1):
            x = estimator.fit_transform_one(x, y)
        return self.final_estimator.fit_predict_proba_one(x, y)

    def draw(self):
        """Draws the pipeline using the ``graphviz`` library."""

        if not GRAPHVIZ_INSTALLED:
            raise ImportError('graphviz is not installed')

        graph = graphviz.Digraph()

        steps = iter(['input'] + list(self.keys()) + ['output'])

        def draw_step(previous_steps=None):

            step = next(steps, None)
            if step is None:
                return

            if isinstance(self.get(step), union.TransformerUnion):
                step = self[step].keys()
            else:
                step = [step]

            for substep in step:
                graph.node(substep)

                for previous_step in previous_steps or []:
                    graph.edge(previous_step, substep)

            # Recurse
            draw_step(step)

        draw_step()

        return graph
