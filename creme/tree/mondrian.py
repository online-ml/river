import abc
import collections
import copy
import functools
import itertools
import math

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False
try:
    from matplotlib import patches
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False
from sklearn import utils

from .. import base
from .. import dist


class Segment:
    """A Segment represents a straight line in space."""

    def __init__(self, lower=math.inf, upper=-math.inf):
        self.lower = lower
        self.upper = upper

    def update(self, x):
        """Updates the lower and upper bounds given a new value.

        Example:

            >>> segment = Segment()
            >>> segment
            [inf, -inf]

            >>> segment.update(-3)
            [-3, -3]

            >>> segment.update(42)
            [-3, 42]

        """
        self.lower = min(self.lower, x)
        self.upper = max(self.upper, x)
        return self

    @property
    def length(self):
        return self.upper - self.lower

    def __repr__(self):
        return f'[{self.lower}, {self.upper}]'

    def __or__(self, other):
        """Returns the union with another ``Segment``.

        Example:

            >>> Segment(-1, 2) | Segment(-2, 1)
            [-2, 2]

        """
        return Segment(min(self.lower, other.lower), max(self.upper, other.upper))

    def __and__(self, other):
        """Returns the intersection with another ``Segment``.

        Example:

            >>> Segment(-1, 2) & Segment(-2, 1)
            [-1, 1]

        """
        return Segment(max(self.lower, other.lower), min(self.upper, other.upper))

    def __lt__(self, x):
        """Returns whether or not a value ``x`` is to the right of the ``Segment``.

        Example:

            >>> Segment(-1, 2) < 3
            True

            >>> Segment(-1, 2) < 0
            False

        """
        return self.upper < x

    def __gt__(self, x):
        """Returns whether or not a value ``x`` is to the left of the ``Segment``.

        Example:

            >>> -2 < Segment(-1, 2)
            True

            >>> 0 < Segment(-1, 2)
            False

        """
        return x < self.lower


class Split:

    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

    def __str__(self):
        return f'{self.feature} < {self.threshold:.3f}'


class Node(abc.ABC):
    """A node is an element of a tree and represents a subspace."""

    def __init__(self, box, split_time, target_dist, n_samples):
        self.box = box
        self.split_time = split_time
        self.target_dist = target_dist
        self.n_samples = n_samples

    @property
    def height(self):
        """Returns the length from here to the furthest leaf."""
        if isinstance(self, Branch):
            return 1 + max(self.left.height, self.right.height)
        return 0

    @property
    def size(self):
        """Returns the number of nodes under here."""
        if isinstance(self, Branch):
            return 1 + self.left.size + self.right.size
        return 1

    def extend(self, x, y, lifetime, parent_split_time, min_samples_split, depth, max_depth,
               random_state):
        """Updates a Node given a new observation."""

        # Step 1: Calculate by how much x is outside of the bounding box
        extent = {}
        e_l, e_u = {}, {}
        total_extent = 0

        for i, xi in x.items():
            e_l[i] = max((self.box[i].lower if i in self.box else xi) - xi, 0)
            e_u[i] = max(xi - (self.box[i].upper if i in self.box else xi), 0)
            extent[i] = e_l[i] + e_u[i]
            total_extent += extent[i]

        # Sample E from an exponential distribution
        E = random_state.exponential(scale=1 / total_extent) if total_extent > 0 else math.inf

        # Determine if a split should happen or not
        if parent_split_time + E < self.split_time and \
           self.n_samples + 1 >= min_samples_split and \
           depth < max_depth:

            # Choose which feature to split on
            names = list(extent.keys())
            weights = [extent[name] / total_extent for name in names]
            feature = random_state.choice(a=names, p=weights)

            # Sample a split threshold
            if x[feature] < self.box[feature]:
                threshold = random_state.uniform(low=x[feature], high=self.box[feature].lower)
            else:
                threshold = random_state.uniform(low=self.box[feature].upper, high=x[feature])

            # Create a new leaf containing the current sample
            new_leaf_box = {i: Segment(xi, xi) for i, xi in x.items()}
            new_leaf = Leaf(
                box=new_leaf_box,
                split_time=lifetime,
                target_dist=self.target_dist.__class__().update(y),
                n_samples=1
            )

            # Decide how to arrange the two new leaves
            if x[feature] < self.box[feature]:
                left, right = new_leaf, self
            else:
                left, right = self, new_leaf

            return Branch(
                split=Split(feature, threshold),
                left=left,
                right=right,
                box={i: new_leaf_box[i] | self.box[i] for i in x},
                split_time=parent_split_time + E,
                target_dist=copy.copy(self.target_dist).update(y),
                n_samples=self.n_samples + 1
            )

        # Update the limits of the Node's box
        for i, xi in x.items():
            self.box[i].update(xi)

        # Update the distribution of the target
        self.target_dist.update(y)

        # Increment the number of observed samples
        self.n_samples += 1

        # Recurse down the tree
        if isinstance(self, Branch):
            kwargs = dict(
                x=x,
                y=y,
                lifetime=lifetime,
                parent_split_time=self.split_time,
                min_samples_split=min_samples_split,
                depth=depth + 1,
                max_depth=max_depth,
                random_state=random_state
            )
            if x[self.split.feature] < self.split.threshold:
                self.left = self.left.extend(**kwargs)
            else:
                self.right = self.right.extend(**kwargs)

        return self

    def predict(self, x, target_dist, parent_split_time, p_nsy):
        """Passes an observation through the tree and updates the target distribution."""

        split_time_diff = self.split_time - parent_split_time

        total_extent = 0
        for i, xi in x.items():
            total_extent += max(xi - (self.box[i].upper if i in self.box else xi), 0)
            total_extent += max((self.box[i].lower if i in self.box else xi) - xi, 0)

        weight = p_nsy

        if isinstance(self, Branch):
            p_js = 1 - math.exp(-split_time_diff * total_extent)
            weight *= p_js

            kwargs = dict(
                x=x,
                target_dist=target_dist,
                parent_split_time=self.split_time,
                p_nsy=p_nsy * (1 - p_js)
            )
            if x[self.split.feature] < self.split.threshold:
                target_dist = self.left.predict(**kwargs)
            else:
                target_dist = self.right.predict(**kwargs)

        # Update the target distribution using the weight
        target_dist += weight * self.target_dist

        return target_dist

    def to_dot(self):

        if not GRAPHVIZ_INSTALLED:
            raise RuntimeError('graphviz is not installed')

        dot = graphviz.Digraph()

        def add_node(node, code):

            # Draw current node
            if isinstance(node, Leaf):
                dot.node(code, f'y = {node.target_dist.mode()}\n' +
                               f'n_samples = {node.n_samples}\n' +
                               f'split_time = {node.split_time:.3f}')
            else:
                dot.node(code, f'{node.split}\n' +
                               f'y = {node.target_dist.mode()}\n' +
                               f'n_samples = {node.n_samples}\n' +
                               f'split_time = {node.split_time:.3f}')
                add_node(node.left, f'{code}0')
                add_node(node.right, f'{code}1')

            # Draw link with the previous node
            is_root = len(code) == 1
            if not is_root:
                dot.edge(code[:-1], code)

        add_node(self, '0')

        return dot

    def plot(self, x=None, y=None, ax=None, colors=None, alpha=1):

        if not MATPLOTLIB_INSTALLED:
            raise RuntimeError('matplotlib is not installed')

        if x is None and y is None:
            raise ValueError('At least one of x and y must be provided')

        ax = ax or plt.axes()
        colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = itertools.cycle(colors)

        def plot_node(node):

            if isinstance(node, Branch):
                plot_node(node.left)
                plot_node(node.right)
                return

            x_segment = node.box.get(x)
            y_segment = node.box.get(y)
            style = dict(color=next(colors), alpha=alpha)

            # If both the x-axis and the y-axis are given then draw a box
            if x_segment is not None and y_segment is not None:
                ax.add_patch(patches.Rectangle(
                    xy=(x_segment.lower, y_segment.lower),
                    width=x_segment.length,
                    height=y_segment.length,
                    **style
                ))
                return

            # If only the x-axis is given then draw a vertical band
            if x_segment is not None:
                ax.axvspan(xmin=x_segment.lower, xmax=x_segment.upper, **style)

            # If only the y-axis is given then draw a horizontal line
            if y_segment is not None:
                ax.axhspan(ymin=y_segment.lower, ymax=y_segment.upper, **style)
                return

        plot_node(self)
        ax.autoscale_view()

        return ax


class Branch(Node):
    """A Branch is a Node that has two children that are also Nodes."""

    def __init__(self, split, left, right, **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.left = left
        self.right = right


class Leaf(Node):
    """A Leaf is a Node that has no children."""


class MondrianTree:

    def __init__(self, target_dist, lifetime, max_depth, min_samples_split, random_state):
        self.tree = Leaf(
            box=collections.defaultdict(functools.partial(Segment, math.inf, -math.inf)),
            split_time=lifetime,
            target_dist=copy.copy(target_dist),
            n_samples=0
        )
        self.lifetime = lifetime
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = utils.check_random_state(random_state)

    def fit_one(self, x, y):

        # Start by getting the prediction from the tree
        y_pred = self.predict_one(x)

        # Update the tree
        self.tree = self.tree.extend(
            x=x,
            y=y,
            lifetime=self.lifetime,
            parent_split_time=0,
            min_samples_split=self.min_samples_split,
            depth=0,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        return y_pred

    def predict_one(self, x):
        target_dist = self.tree.predict(
            x=x,
            target_dist=self.tree.target_dist.__class__(),
            parent_split_time=0,
            p_nsy=1
        )
        return target_dist.mode()


class MondrianTreeRegressor(MondrianTree, base.Regressor):
    """Mondrian tree for regression.

    Example:

        >>> from creme import compose
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import optim
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from creme import tree
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_boston,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('mondrian', tree.MondrianTreeRegressor(random_state=42))
        ... ])
        >>> metric = metrics.MAE()

        >>> model_selection.online_score(X_y, model, metric)
        MAE: 4.250497

    References:

    1. `Mondrian Forests: Efficient Online Random Forests <https://github.com/balajiln/mondrianforest>`_
    2. `Decision Trees and Forests: A Probabilistic Perspective <http://www.gatsby.ucl.ac.uk/~balaji/balaji-phd-thesis.pdf>`_
    3. `scikit-garden <https://github.com/scikit-garden/scikit-garden/tree/master/skgarden/mondrian>`_

    """

    def __init__(self, lifetime=1, max_depth=8, min_samples_split=1, random_state=None):
        return super().__init__(
            target_dist=dist.Normal(),
            lifetime=lifetime,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )


class MondrianTreeClassifier(MondrianTree, base.MultiClassifier):
    """Mondrian tree for classification.

    Example:

        >>> from creme import compose
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from creme import tree
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     load_dataset=datasets.load_breast_cancer,
        ...     shuffle=True,
        ...     random_state=42
        ... )
        >>> model = compose.Pipeline([
        ...     ('scale', preprocessing.StandardScaler()),
        ...     ('mondrian', tree.MondrianTreeClassifier(random_state=42))
        ... ])
        >>> metric = metrics.F1Score()

        >>> model_selection.online_score(X_y, model, metric)
        F1Score: 0.822309

    References:

    1. `Mondrian Forests: Efficient Online Random Forests <https://github.com/balajiln/mondrianforest>`_
    2. `Decision Trees and Forests: A Probabilistic Perspective <http://www.gatsby.ucl.ac.uk/~balaji/balaji-phd-thesis.pdf>`_
    3. `scikit-garden <https://github.com/scikit-garden/scikit-garden/tree/master/skgarden/mondrian>`_

    """

    def __init__(self, lifetime=1, max_depth=5, min_samples_split=1, random_state=None):
        return super().__init__(
            target_dist=dist.Multinomial(),
            lifetime=lifetime,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def predict_proba_one(self, x):
        target_dist = self.tree.predict(
            x=x,
            target_dist=self.tree.target_dist.__class__(),
            parent_split_time=0,
            p_nsy=1
        )
        y_pred = target_dist.counter
        y_pred_sum = sum(y_pred.values())
        return {c: y / y_pred_sum for c, y in y_pred.items()}
