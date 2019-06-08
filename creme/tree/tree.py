import collections
import functools

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from .. import base
from .. import utils

from . import branch
from . import criterion
from . import leaf


CRITERIA_CLF = {'gini': criterion.gini_impurity, 'entropy': criterion.entropy}


class DecisionTreeClassifier(base.MultiClassClassifier):
    """

    Parameters:
        max_bins (int): Maximum number of bins used for discretizing continuous values when using
            `utils.Histogram`.

    Attributes:
        histograms (collections.defaultdict)

    """

    def __init__(self, criterion='gini', max_depth=5, min_samples_split=10, patience=10,
                 max_bins=30):
        self.criterion = CRITERIA_CLF[criterion]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.patience = patience
        self.max_bins = max_bins
        self.delta = 0.1
        self.bound_threshold = 0.05
        self.histograms = collections.defaultdict(functools.partial(
            utils.Histogram,
            max_bins=max_bins
        ))
        self.root = leaf.Leaf(depth=0, tree=self)

    def fit_one(self, x, y):
        self.root = self.root.update(x, y)
        return self

    def predict_proba_one(self, x):
        l = self.root.get_leaf(x)
        return {
            label: count / l.n_samples
            for label, count in l.class_counts.items()
        }

    def to_dot(self):
        """Returns a GraphViz representation of the decision tree."""

        if not GRAPHVIZ_INSTALLED:
            raise RuntimeError('graphviz is not installed')

        dot = graphviz.Digraph()

        def add_node(node, code):

            # Draw the current node
            if isinstance(node, leaf.Leaf):
                dot.node(code, str(node.class_counts))
            else:
                dot.node(code, str(node.split))
                add_node(node.left, f'{code}0')
                add_node(node.right, f'{code}1')

            # Draw the link with the previous node
            is_root = len(code) == 1
            if not is_root:
                dot.edge(code[:-1], code)

        add_node(self.root, '0')

        return dot

    def debug_one(self, x):
        """Prints an explanation of how ``x`` is predicted."""
        node = self.root

        while isinstance(node, branch.Branch):
            if node.split.test(x):
                print('not', node.split)
                node = node.left
            else:
                print(node.split)
                node = node.right

        print(node.class_counts)
