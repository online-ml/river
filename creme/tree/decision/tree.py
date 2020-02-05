import abc
import collections
import functools
import numbers

from ... import base
from ... import proba

from . import criteria
from . import leaf
from . import splitting


CRITERIA_CLF = {'gini': criteria.gini_impurity, 'entropy': criteria.entropy}


class BaseDecisionTree(abc.ABC):

    def __init__(self, criterion='gini', patience=250, max_depth=5, min_split_gain=0.,
                 min_child_samples=20, confidence=1e-10, tie_threshold=5e-2, n_split_points=30,
                 max_bins=60):
        self.criterion = CRITERIA_CLF[criterion]
        self.patience = patience
        self.max_depth = max_depth
        self.min_split_gain = min_split_gain
        self.min_child_samples = min_child_samples
        self.confidence = confidence
        self.tie_threshold = tie_threshold
        self.n_split_points = n_split_points
        self.max_bins = max_bins

        self.root = leaf.Leaf(depth=0, tree=self, target_dist=proba.Multinomial())

    def fit_one(self, x, y):
        self.root = self.root.update(x, y)
        return self

    @abc.abstractmethod
    def _get_split_enum(self, typ):
        """Returns the appropriate split enumerator for a given type."""

    def draw(self):
        """Returns a GraphViz representation of the decision tree."""

        import graphviz

        dot = graphviz.Digraph()

        def add_node(node, path):

            if isinstance(node, leaf.Leaf):
                # Draw a leaf
                dot.node(path, str(node.target_dist), shape='box')
            else:
                # Draw a branch
                dot.node(path, str(node.split))
                add_node(node.left, f'{path}0')
                add_node(node.right, f'{path}1')

            # Draw the link with the previous node
            is_root = len(path) == 1
            if not is_root:
                dot.edge(path[:-1], path)

        add_node(node=self.root, path='0')

        return dot

    def debug_one(self, x, **print_params):
        """Prints an explanation of how ``x`` is predicted.

        Parameters:
            x (dict)
            **print_params (dict): Parameters passed to the `print` function.

        """
        node = self.root
        _print = functools.partial(print, **print_params)

        for node in self.root.path(x):
            if isinstance(node, leaf.Leaf):
                _print(node.target_dist)
                break
            if node.split(x):
                _print('not', node.split)
            else:
                _print(node.split)


class DecisionTreeClassifier(BaseDecisionTree, base.MultiClassifier):
    """Decision tree classifier.

    Parameters:
        criterion (str): The function to measure the quality of a split. Set to ``'gini'`` in order
            to use Gini impurity and ``'entropy'`` for information gain.
        patience (int): Time to wait between split attempts.
        max_depth (int): Maximum tree depth.
        min_split_gain (float): Minimum impurity gain required to make a split eligible.
        min_child_samples (int): Minimum number of data needed in a leaf.
        confidence (float): Threshold used to compare with the Hoeffding bound.
        tie_threshold (float): Threshold to handle ties between equally performing attributes.
        n_split_points (int): Number of split points considered for splitting numerical variables.
        max_bins (int): Number of histogram bins used for approximating the distribution of
            numerical variables.

    Attributes:
        root (Leaf)

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import tree

            >>> X_y = datasets.Phishing()

            >>> model = tree.DecisionTreeClassifier(
            ...     patience=100,
            ...     confidence=1e-5,
            ...     criterion='gini'
            ... )

            >>> metric = metrics.LogLoss()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            LogLoss: 0.701038

    References:
        .. [1] `Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 71-80). <https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf>`_
        .. [2] `Article by The Morning Paper <https://blog.acolyer.org/2015/08/26/mining-high-speed-data-streams/>`_

    """

    def _get_split_enum(self, value):
        """Returns an appropriate split enumerator given a feature's type."""
        if isinstance(value, numbers.Number):
            return splitting.HistSplitEnum(n_bins=self.max_bins, n_splits=self.n_split_points)

        elif isinstance(value, (str, bool)):
            return splitting.CategoricalSplitEnum()

        raise ValueError(f'The type of {value} ({type(value)}) is not supported')

    def predict_proba_one(self, x):
        leaf = collections.deque(self.root.path(x), maxlen=1).pop()
        return leaf.predict(x)
