try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from .. import base
from .. import proba

from . import criteria
from . import leaf


CRITERIA_CLF = {'gini': criteria.gini, 'entropy': criteria.entropy}


class DecisionTreeClassifier(base.MultiClassifier):
    """Decision tree classifier.

    Parameters:
        criterion (str): The function to measure the quality of a split. Set to ``'gini'`` in order
        to use Gini impurity and ``'entropy'`` for information gain.
        patience (int): Time to wait between split attempts.
        max_depth (int): Maximum tree depth.
        min_child_samples (int): Minimum number of data needed in a leaf.
        confidence (float): Threshold used to compare with the Hoeffding bound.
        tie_threshold (float): Threshold to handle ties between equally performing attributes.

    Attributes:
        root

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import tree

            >>> X_y = datasets.fetch_electricity()

            >>> model = tree.DecisionTreeClassifier(
            ...     patience=200,
            ...     criterion='gini'
            ... )

            >>> metric = metrics.ROCAUC()

            >>> model_selection.online_score(X_y, model, metric)
            ROCAUC: 0.80453

    References:
        1. `Mining High-Speed Data Streams <https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf>`_
        2. `The Morning Paper <https://blog.acolyer.org/2015/08/26/mining-high-speed-data-streams/>`_

    """

    def __init__(self, criterion='entropy', patience=10, max_depth=5, min_child_samples=20,
                 confidence=1e-5, tie_threshold=5e-2):
        self.criterion = CRITERIA_CLF[criterion]
        self.patience = patience
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples

        self.confidence = confidence
        self.tie_threshold = tie_threshold
        self.root = leaf.Leaf(depth=0, tree=self, target_dist=proba.Multinomial())

    def fit_one(self, x, y):
        self.root = self.root.update(x, y)
        return self

    def predict_proba_one(self, x):
        return self.root.get_leaf(x).predict(x)

    def draw(self):
        """Returns a GraphViz representation of the decision tree."""

        if not GRAPHVIZ_INSTALLED:
            raise RuntimeError('graphviz is not installed')

        dot = graphviz.Digraph()

        def add_node(node, path):

            # Draw the current node
            if isinstance(node, leaf.Leaf):
                dot.node(path, str(node.target_dist))
            else:
                dot.node(path, str(node.split))
                add_node(node.left, f'{path}0')
                add_node(node.right, f'{path}1')

            # Draw the link with the previous node
            is_root = len(path) == 1
            if not is_root:
                dot.edge(path[:-1], path)

        add_node(node=self.root, path='0')

        return dot

    def debug_one(self, x):
        """Prints an explanation of how ``x`` is predicted."""
        node = self.root

        while isinstance(node, leaf.Branch):
            if node.split.test(x):
                print('not', node.split)
                node = node.left
            else:
                print(node.split)
                node = node.right

        print(node.class_counts)
