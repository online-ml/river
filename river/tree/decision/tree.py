import abc
import collections
import functools
import io
import itertools
import math
import numbers
import typing

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from river import base
from river import proba

from . import criteria
from . import leaf
from . import splitting

CRITERIA_CLF = {'gini': criteria.gini_impurity, 'entropy': criteria.entropy}


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..., (s3,None)

    We use this to iterate over successive pairs of nodes in a path for curtailment purposes.

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b)


class BaseDecisionTree(abc.ABC):

    def __init__(self, criterion='gini', patience=250, max_depth=5, min_split_gain=0.,
                 min_child_samples=20, confidence=1e-10, tie_threshold=5e-2, n_split_points=30,
                 max_bins=60, curtail_under=10):
        self.criterion = criterion
        self.patience = patience
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_child_samples = min_child_samples
        self.confidence = confidence
        self.tie_threshold = tie_threshold
        self.n_split_points = n_split_points
        self.max_bins = max_bins
        self.curtail_under = curtail_under

        self.criterion_func = CRITERIA_CLF[criterion]
        self.root = leaf.Leaf(depth=0, tree=self, target_dist=proba.Multinomial())

    def learn_one(self, x, y):
        self.root = self.root.update(x, y)
        return self

    @abc.abstractmethod
    def _make_leaf_dist(self):
        """Returns a target distribution for a newly instantiated leaf."""

    @abc.abstractmethod
    def _get_split_enum(self, value):
        """Returns the appropriate split enumerator for a given value based on it's type."""

    def draw(self, max_depth: int = None):
        """Draws the tree using the `graphviz` library.

        Parameters
        ----------
        max_depth
            Only the root will be drawn when set to `0`. Every node will be drawn when
            set to `None`.

        Examples
        --------

        >>> from river import datasets
        >>> from river import tree

        >>> model = tree.DecisionTreeClassifier(
        ...    patience=10,
        ...    confidence=1e-5,
        ...    criterion='gini',
        ...    max_depth=10,
        ...    tie_threshold=0.05,
        ...    min_child_samples=0,
        ... )

        >>> for x, y in datasets.Phishing():
        ...    model = model.learn_one(x, y)

        >>> dot = model.draw()

        .. image:: /img/dtree_draw.svg
            :align: center

        """

        if max_depth is None:
            max_depth = math.inf

        dot = graphviz.Digraph(
            graph_attr={'splines': 'ortho'},
            node_attr={
                'shape': 'box', 'penwidth': '1.2', 'fontname': 'trebuchet',
                'fontsize': '11', 'margin': '0.1,0.0'
            },
            edge_attr={'penwidth': '0.6', 'center': 'true'}
        )

        # Do a first pass to guess the number of classes
        n_classes = len(set(itertools.chain(*[
            list(node.target_dist)
            for node, _ in self.root.iter_dfs()])
        ))

        # Pick a color palette which maps classes to colors
        new_color = functools.partial(next, iter(_color_brew(n_classes)))
        palette = collections.defaultdict(new_color)

        for parent_no, child_no, _, child, child_depth in self.root.iter_edges():

            if child_depth > max_depth:
                continue

            if isinstance(child, leaf.Branch):
                text = f'{child.split} \n {child.target_dist} \n samples: {child.n_samples}'
            elif isinstance(child, leaf.Leaf):
                text = f'{child.target_dist} \n samples: {child.n_samples}'

            # Pick a color, the hue depends on the class and the transparency on the distribution
            mode = child.target_dist.mode
            if mode is not None:
                p_mode = child.target_dist.pmf(mode)
                alpha = (p_mode - 1 / n_classes) / (1 - 1 / n_classes)
                fillcolor = str(transparency_hex(color=palette[mode], alpha=alpha))
            else:
                fillcolor = '#FFFFFF'

            dot.node(f'{child_no}', text, fillcolor=fillcolor, style='filled')

            if parent_no is not None:
                dot.edge(f'{parent_no}', f'{child_no}')

        return dot

    def debug_one(self, x: dict) -> str:
        """Prints an explanation of how `x` is predicted.

        Parameters
        ----------
        x
            A dictionary of features.

        """

        # We'll redirect all the print statement to a buffer, we'll return the content of the
        # buffer at the end
        buffer = io.StringIO()
        _print = functools.partial(print, file=buffer)

        node = self.root

        for node in self.root.path(x):
            if isinstance(node, leaf.Leaf):
                _print(node.target_dist)
                break
            if node.split(x):
                _print('not', node.split)
            else:
                _print(node.split)

        return buffer.getvalue()


class DecisionTreeClassifier(BaseDecisionTree, base.Classifier):
    """Decision tree classifier.

    Numeric features will be treated as continuous features. You can cast a feature to a string in
    order to treat it as a categorical feature.

    Parameters
    ----------
    criterion
        The function to measure the quality of a split. Set to `'gini'` in order
        to use Gini impurity and `'entropy'` for information gain.
    patience
        Time to wait between split attempts.
    max_depth
        Maximum tree depth.
    min_split_gain
        Minimum impurity gain required to make a split eligible.
    min_child_samples
        Minimum number of data needed in a leaf.
    confidence
        Threshold used to compare with the Hoeffding bound.
    tie_threshold
        Threshold to handle ties between equally performing attributes.
    n_split_points
        Number of split points considered for splitting numerical variables.
    max_bins
        Number of histogram bins used for approximating the distribution of
        numerical variables.
    curtail_under
        Determines the minimum amount of samples for a node to be eligible to
        make predictions. For instance, if a leaf doesn't contain at least `curtail_under`
        samples, then it's parent will be used instead. If said parent also doesn't contain at
        leaf `curtail_under` samples, then it's parent is used, etc. This helps to counter
        the fact that new leaves start with no samples at all, therefore their predictions
        might be unreliable. No curtailment will be applied if you set this to `0`. However,
        note that using even a small amount of curtailment almost always results in better
        performance.

    Attributes
    ----------
    root

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()

    >>> model = tree.DecisionTreeClassifier(
    ...     patience=100,
    ...     confidence=1e-5,
    ...     criterion='gini'
    ... )

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.51755

    References
    ----------
    [^1]: [Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 71-80).](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)
    [^2]: [Article by The Morning Paper](https://blog.acolyer.org/2015/08/26/mining-high-speed-data-streams/)

    """

    @property
    def _multiclass(self):
        return True

    def _make_leaf_dist(self):
        return proba.Multinomial()

    def _get_split_enum(self, value):
        """Returns an appropriate split enumerator given a feature's type."""
        if isinstance(value, numbers.Number):
            return splitting.HistSplitEnum(n_bins=self.max_bins, n_splits=self.n_split_points)

        elif isinstance(value, (str, bool)):
            return splitting.CategoricalSplitEnum()

        raise ValueError(f'The type of {value} ({type(value)}) is not supported')

    def predict_proba_one(self, x):

        # Find the deepest node which contains at least curtail_under samples
        for node, child in pairwise(self.root.path(x)):
            if child is None or child.n_samples < self.curtail_under:
                break

        return {c: node.target_dist.pmf(c) for c in node.target_dist}


def _color_brew(n: int) -> typing.List[typing.Tuple[int, int, int]]:
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n
        The number of colors required.

    Returns
    -------
    List of n tuples of form (R, G, B) being the components of each color.

    References
    ----------
    [^1]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_export.py

    """

    colors = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = .75, .9
    c = s * v
    m = v - c

    for h in [i for i in range(25, 385, int(360 / n))]:

        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))

        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]

        # Shift the initial RGB values to match value and store
        colors.append((
            (int(255 * (r + m))),
            (int(255 * (g + m))),
            (int(255 * (b + m)))
        ))

    return colors


def transparency_hex(color: typing.Tuple[int, int, int], alpha: float) -> str:
    """Apply alpha coefficient on hexadecimal color."""
    return '#%02x%02x%02x' % tuple([int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color])
