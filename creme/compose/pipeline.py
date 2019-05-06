import collections
import itertools

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from sklearn.utils import metaestimators

from . import func
from . import union


__all__ = ['Pipeline']


class Pipeline(collections.OrderedDict):
    """Chains a sequence of estimators.

    Sequentially apply a list of estimators. Pipelines helps to define machine learning systems in a
    declarative style, which makes a lot of sense when we think in a stream manner. For further
    information and practical examples, take a look at the
    `user guide <../notebooks/the-art-of-using-pipelines.html>`_.

    Parameters:
        steps (list): Ideally a list of (name, estimator) tuples. If an estimator is given without
            a name then a name is automatically inferred from the estimator.

    Example:

        ::

            >>> from creme import compose
            >>> from creme import feature_extraction
            >>> from creme import linear_model
            >>> from creme import preprocessing

            >>> tfidf = feature_extraction.TFIDFVectorizer('text')
            >>> counts = feature_extraction.CountVectorizer('text')
            >>> text_part = compose.Whitelister('text') | (tfidf + counts)

            >>> num_part = compose.Whitelister(['a', 'b']) | preprocessing.PolynomialExtender()

            >>> model = text_part + num_part
            >>> model |= preprocessing.StandardScaler()
            >>> model |= linear_model.LinearRegression()

            >>> dot = model.draw()

        .. image:: ../_static/pipeline_docstring.svg
            :align: center

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

    def __add__(self, other):
        """Merges with another Pipeline or TransformerUnion into a TransformerUnion."""
        if isinstance(other, union.TransformerUnion):
            return other.__add__(self)
        return union.TransformerUnion([self, other])

    def __radd__(self, other):
        """Merges with another Pipeline or TransformerUnion into a TransformerUnion."""
        if isinstance(other, union.TransformerUnion):
            return other.__add__(self)
        return union.TransformerUnion([other, self])

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
        x_transformed = x

        for transformer in itertools.islice(self.values(), len(self) - 1):
            x_transformed = transformer.transform_one(x_transformed)

            # The supervised parts of a TransformerUnion have to be updated
            if isinstance(transformer, union.TransformerUnion):
                for sub_transformer in transformer.values():
                    if sub_transformer.is_supervised:
                        sub_transformer.fit_one(x, y)
                continue

            # If a transformer is supervised then it has to be updated
            if transformer.is_supervised:
                transformer.fit_one(x, y)

        self.final_estimator.fit_one(x_transformed, y)
        return self

    def run_transformers(self, x):
        for transformer in itertools.islice(self.values(), len(self) - 1):

            if isinstance(transformer, union.TransformerUnion):
                for sub_transformer in transformer.values():
                    if not sub_transformer.is_supervised:
                        sub_transformer.fit_one(x)
                x = transformer.transform_one(x)
                continue

            if not transformer.is_supervised:
                transformer.fit_one(x)
            x = transformer.transform_one(x)

        return x

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def transform_one(self, x):
        """Transform an input.

        Only works if each estimator has a ``transform_one`` method.

        """
        x = self.run_transformers(x)
        final_tranformer = self.final_estimator
        if not final_tranformer.is_supervised:
            final_tranformer.fit_one(x)
        x = final_tranformer.transform_one(x)
        return x

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_one(self, x):
        """Predict output.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a
        ``predict_one`` method.

        """
        x = self.run_transformers(x)
        return self.final_estimator.predict_one(x)

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_proba_one(self, x):
        """Predicts probabilities.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a
        ``predict_proba_one`` method.

        """
        x = self.run_transformers(x)
        return self.final_estimator.predict_proba_one(x)

    def draw(self):
        """Draws the pipeline using the ``graphviz`` library."""

        if not GRAPHVIZ_INSTALLED:
            raise ImportError('graphviz is not installed')

        def get_first_estimator(d):
            """Gets first estimator key of a Pipeline or TransformerUnion."""

            for first_key in d.keys():
                first_step = d.get(first_key)
                break

            if isinstance(first_step, (Pipeline, union.TransformerUnion)):
                # Recurse
                first_key = get_first_estimator(first_step)

            return first_key

        def draw_step(node, previous_node):
            """Draws a node and its previous edge."""
            if node in nodes:
                node = node + '_'
                return draw_step(node, previous_node)

            graph.node(node, node.rstrip('_'))
            graph.edge(previous_node, node)
            nodes.append(node)
            edges.append(previous_node)

        def draw_steps(d=self, skip_first=False):
            """Draws all estimators graph nodes and edges."""

            union_ending_node_ix = None

            for key in list(d.keys()):

                if skip_first:
                    skip_first = False
                    continue

                step = d.get(key)

                # If step is a Pipeline recurse on step
                if isinstance(step, Pipeline):
                    draw_steps(step)

                # If step is a TransformerUnion, dive inside
                elif isinstance(step, union.TransformerUnion):

                    node_before_union = nodes[-1]

                    # Draw each TransformerUnion steps
                    for ix, sub_key in enumerate(step.keys()):

                        sub_step = step.get(sub_key)

                        # If sub step is another nested step, draw its first estimator and recurse
                        if isinstance(sub_step, (Pipeline, union.TransformerUnion)):
                            sub_sub_key = get_first_estimator(sub_step)
                            draw_step(node=sub_sub_key, previous_node=node_before_union)
                            draw_steps(d=sub_step, skip_first=True)
                        # Else just draw it
                        else:
                            draw_step(node=sub_key, previous_node=nodes[-1 - ix])

                    union_ending_node_ix = len(nodes)

                else:
                    draw_step(key, nodes[-1])

                # If previous step was a TransformerUnion and following node have been drawn
                if union_ending_node_ix == len(nodes) - 1:
                    # Connect TransformerUnion child nodes with the next step
                    for node in nodes[1: -1]:
                        if node not in edges:
                            graph.edge(node, nodes[union_ending_node_ix])
                            edges.append(node)
                    # Reset TransformerUnion flag
                    union_ending_node_ix = None

        nodes, edges = ['input'], []
        graph = graphviz.Digraph()
        graph.node('input')

        draw_steps()

        graph.node('output')
        graph.edge(nodes[-1], 'output')

        return graph
