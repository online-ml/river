import collections
import itertools
import types

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

            >>> num_part = compose.Whitelister('a', 'b') | preprocessing.PolynomialExtender()

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

    @property
    def transformers(self):
        """If a pipeline has $n$ steps, then the first $n-1$ are necessarily transformers."""
        if isinstance(self.final_estimator, base.Transformer):
            return self.values()
        return itertools.islice(self.values(), len(self) - 1)

    @property
    def is_supervised(self):
        """Only works if all the steps of the pipelines are transformers."""
        return any(transformer.is_supervised for transformer in self.values())

    def add_step(self, step, at_start):
        """Adds a step to either end of the pipeline while taking care of the input type."""

        # Infer a name if none is given
        if not isinstance(step, (list, tuple)):
            step = (str(step), step)

        name, estimator = step

        # If a function is given then wrap it in a FuncTransformer
        if isinstance(estimator, types.FunctionType):
            name = estimator.__name__
            estimator = func.FuncTransformer(estimator)

        # Check if an identical step has already been inserted
        if name in self:
            raise KeyError(f'{name} already exists')

        # Instantiate the estimator if it hasn't been done
        if isinstance(estimator, type):
            estimator = estimator()

        # Store the step
        self[name] = estimator

        # Move the step to the start of the pipeline if so instructed
        if at_start:
            self.move_to_end(step[0], last=False)

    @property
    def final_estimator(self):
        """The final estimator."""
        return self[next(reversed(self))]

    def fit_one(self, x, y=None):
        """Fits each step with ``x``."""

        # Loop over the first n - 1 steps, which should all be transformers
        for t in itertools.islice(self.values(), len(self) - 1):
            x_pre = x
            x = t.transform_one(x)

            # If a transformer is supervised then it has to be updated
            if t.is_supervised:

                if isinstance(t, union.TransformerUnion):
                    for sub_t in t.values():
                        if sub_t.is_supervised:
                            sub_t.fit_one(x_pre, y)

                else:
                    t.fit_one(x_pre, y)

        self.final_estimator.fit_one(x, y)
        return self

    def transform_one(self, x):
        """Transform an input.

        Only works if each estimator has a ``transform_one`` method.

        """
        for transformer in self.transformers:

            if isinstance(transformer, union.TransformerUnion):

                # Fit the unsupervised part of the union
                for sub_transformer in transformer.values():
                    if not sub_transformer.is_supervised:
                        sub_transformer.fit_one(x)

            elif not transformer.is_supervised:
                transformer.fit_one(x)

            x = transformer.transform_one(x)

        return x

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_one(self, x):
        """Predict output.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a
        ``predict_one`` method.

        """
        x = self.transform_one(x)
        return self.final_estimator.predict_one(x)

    @metaestimators.if_delegate_has_method(delegate='final_estimator')
    def predict_proba_one(self, x):
        """Predicts probabilities.

        Only works if each estimator has a ``transform_one`` method and the final estimator has a
        ``predict_proba_one`` method.

        """
        x = self.transform_one(x)
        return self.final_estimator.predict_proba_one(x)

    def debug_one(self, x, show_types=True):
        """Displays the state of a set of features as it goes through the pipeline.

        Parameters:
            x (dict) A set of features.
            show_types (bool): Whether or not to display the type of feature along with it's value.

        """
        def print_features(x, indent=False, space_after=True):
            for k, v in x.items():
                type_str = f' ({type(v).__name__})' if show_types else ''
                print(('\t' if indent else '') + f'{k}: {v}' + type_str)
            if space_after:
                print()

        def print_title(title, indent=False):
            print(('\t' if indent else '') + title)
            print(('\t' if indent else '') + '-' * len(title))

        # Print the initial state of the features
        print_title('0. Input')
        print_features(x)

        for i, t in enumerate(self.transformers):
            if isinstance(t, union.TransformerUnion):
                print_title(f'{i+1}. Transformer union')
                for j, (name, sub_t) in enumerate(t.items()):
                    print_title(f'{i+1}.{j} {name}', indent=True)
                    print_features(sub_t.transform_one(x), indent=True)
                x = t.transform_one(x)
                print_features(x)
            else:
                print_title(f'{i+1}. {t}')
                x = t.transform_one(x)
                print_features(x)

        # Print the predicted output from the final estimator
        final = self.final_estimator
        if not isinstance(final, base.Transformer):
            print_title(f'{len(self)}. {final}')
            if isinstance(final, base.Classifier):
                print_features(final.predict_proba_one(x), space_after=False)
            else:
                print(final.predict_one(x))

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

            for name, step in d.items():

                if skip_first:
                    skip_first = False
                    continue

                # If step is a Pipeline recurse on step
                if isinstance(step, Pipeline):
                    draw_steps(step)

                # If step is a TransformerUnion, dive inside
                elif isinstance(step, union.TransformerUnion):

                    node_before_union = nodes[-1]

                    # Draw each TransformerUnion steps
                    for sub_name, sub_step in step.items():

                        # If sub step is another nested step, draw its first estimator and recurse
                        if isinstance(sub_step, (Pipeline, union.TransformerUnion)):
                            sub_sub_key = get_first_estimator(sub_step)
                            draw_step(node=sub_sub_key, previous_node=node_before_union)
                            draw_steps(d=sub_step, skip_first=True)
                        # Else just draw it
                        else:
                            draw_step(node=sub_name, previous_node=node_before_union)

                    union_ending_node_ix = len(nodes)

                else:
                    draw_step(name, nodes[-1])

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
