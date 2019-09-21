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


class Network(collections.UserList):
    """An abstraction to help with drawing pipelines."""

    def __init__(self, nodes, links, directed, name=None):
        super().__init__()
        for node in nodes:
            self.append(node)
        self.links = set()
        for link in links:
            self.link(*link)
        self.directed = directed
        self.name = name

    def append(self, a):
        if a not in self:
            super().append(a)

    def link(self, a, b):
        self.append(a)
        self.append(b)
        self.links.add((self.index(a), self.index(b)))

    def draw(self):
        dot = graphviz.Digraph()

        def draw_node(a):
            if isinstance(a, Network):
                for part in a:
                    draw_node(part)
            else:
                dot.node(a)

        for a in self:
            draw_node(a)

        def draw_link(a, b):

            if isinstance(a, Network):
                # Connect the last part of a with b
                if a.directed:
                    draw_link(a[-1], b)
                # Connect each part of a with b
                else:
                    for part in a:
                        draw_link(part, b)

            elif isinstance(b, Network):
                # Connect the first part of b with a
                if b.directed:

                    if b.name is not None:
                        # If the graph has a name, then we treat is as a cluster
                        c = b.draw()
                        c.attr(label=b.name)
                        c.name = f'cluster_{b.name}'
                        dot.subgraph(c)
                    else:
                        dot.subgraph(b.draw())

                    draw_link(a, b[0])
                # Connect each part of b with a
                else:
                    for part in b:
                        draw_link(a, part)

            else:
                dot.edge(a, b)

        for a, b in self.links:
            draw_link(self[a], self[b])

        return dot


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

        The following shows an example of using ``debug_one`` to visualize how the information
        flows and changes throughout the pipeline.

        ::

            >>> from creme import compose
            >>> from creme import feature_extraction
            >>> from creme import naive_bayes
            >>> from sklearn import datasets

            >>> X_y = [
            ...     ('A positive comment', True),
            ...     ('A negative comment', False),
            ...     ('A happy comment', True),
            ...     ('A lovely comment', True),
            ...     ('A harsh comment', False)
            ... ]

            >>> tfidf = feature_extraction.TFIDFVectorizer() | compose.Renamer(prefix='tfidf_')
            >>> counts = feature_extraction.CountVectorizer() | compose.Renamer(prefix='count_')
            >>> mnb = naive_bayes.MultinomialNB()
            >>> model = (tfidf + counts) | mnb

            >>> for x, y in X_y:
            ...     model = model.fit_one(x, y)

            >>> model.debug_one(X_y[0][0])
            0. Input
            --------
            A positive comment
            <BLANKLINE>
            1. Transformer union
            --------------------
                1.0 TFIDFVectorizer | Renamer
                -----------------------------
                tfidf_comment: 0.47606 (float)
                tfidf_positive: 0.87942 (float)
            <BLANKLINE>
                1.1 CountVectorizer | Renamer
                -----------------------------
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

    def __repr__(self):
        return (
            'Pipeline (\n    ' +
            '    '.join(',\n'.join(map(repr, self.values())).splitlines(True)) +
            '\n)'
        )

    @property
    def __class__(self):
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
            self.move_to_end(name, last=False)

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

    def debug_one(self, x, show_types=True, n_decimals=5):
        """Displays the state of a set of features as it goes through the pipeline.

        Parameters:
            x (dict) A set of features.
            show_types (bool): Whether or not to display the type of feature along with it's value.
            n_decimals (int): Number of decimals to display for each floating point value.

        """

        TAB = ' ' * 4

        def format_value(x):
            if isinstance(x, float):
                return '{:.{prec}f}'.format(x, prec=n_decimals)
            return x

        def print_features(x, show_types, indent=False, space_after=True):

            # Some transformers accept strings as input instead of dicts
            if isinstance(x, str):
                print(x)
            else:
                for k, v in sorted(x.items()):
                    type_str = f' ({type(v).__name__})' if show_types else ''
                    print((TAB if indent else '') + f'{k}: {format_value(v)}' + type_str)
            if space_after:
                print()

        def print_title(title, indent=False):
            print((TAB if indent else '') + title)
            print((TAB if indent else '') + '-' * len(title))

        # Print the initial state of the features
        print_title('0. Input')
        print_features(x, show_types=show_types)

        # Print the state of x at each step
        for i, t in enumerate(self.transformers):

            if isinstance(t, union.TransformerUnion):
                print_title(f'{i+1}. Transformer union')
                for j, (name, sub_t) in enumerate(t.items()):
                    print_title(f'{i+1}.{j} {name}', indent=True)
                    print_features(sub_t.transform_one(x), show_types=show_types, indent=True)
                x = t.transform_one(x)
                print_features(x, show_types=show_types)

            else:
                print_title(f'{i+1}. {t}')
                x = t.transform_one(x)
                print_features(x, show_types=show_types)

        # Print the predicted output from the final estimator
        final = self.final_estimator
        if not isinstance(final, base.Transformer):
            print_title(f'{len(self)}. {final}')

            if hasattr(final, 'debug_one'):
                final.debug_one(x)
                print()

            if isinstance(final, base.Classifier):
                print_features(final.predict_proba_one(x), show_types=False, space_after=False)
            else:
                print(final.predict_one(x))

    def draw(self):
        """Draws the pipeline using the ``graphviz`` library."""

        def networkify(step):

            # Unions are converted to an undirected network
            if isinstance(step, union.TransformerUnion):
                return Network(nodes=map(networkify, step.values()), links=[], directed=False)

            # Pipelines are converted to a directed network
            if isinstance(step, Pipeline):
                return Network(
                    nodes=[],
                    links=zip(
                        map(networkify, list(step.values())[:-1]),
                        map(networkify, list(step.values())[1:])
                    ),
                    directed=True
                )

            # Wrapper models are unrolled
            if isinstance(step, base.Wrapper):
                return Network(
                    nodes=[networkify(step._model)],
                    links=[],
                    directed=True,
                    name=type(step).__name__
                )

            # Other steps are treated as strings
            return str(step)

        # Draw input
        net = Network(nodes=['x'], links=[], directed=True)
        previous = 'x'

        # Draw each step
        for step in self.values():
            current = networkify(step)
            net.link(previous, current)
            previous = current

        # Draw output
        net.link(previous, 'y')

        return net.draw()
