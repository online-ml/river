import abc
import collections
import functools
import numbers
import itertools

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from ... import base
from ... import proba

from ..base import Leaf
from ..base import Branch

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

    def draw(self, max_depth=30):
        """Draws the tree using the ``graphviz`` library."""

        dot = graphviz.Digraph(
            graph_attr={'splines': 'ortho'},
            node_attr={'shape': 'box', 'penwidth': '1.2', 'fontname': 'trebuchet',
                    'fontsize': '11', 'margin': '0.1,0.0'},
            edge_attr={'penwidth': '0.6', 'center': 'true'}
        )

        def apply_alpha(color, alpha):
            """Apply alpha coefficient on hexadecimal color."""
            alpha_map = {
                0: '00', 0.05: '0D', 0.10: '1A', 0.15: '26', 0.20: '33', 0.25: '40',
                0.30: '4D', 0.35: '59', 0.40: '66', 0.45: '73', 0.50: '80', 0.55: '8C',
                0.60: '99', 0.65: 'A6', 0.70: 'B3', 0.75: 'BF', 0.80: 'CC', 0.85: 'D9',
                0.90: 'E6', 0.95: 'F2', 1: ''
            }

            alpha = round(alpha * 20) / 20.0

            return f'#{alpha_map[alpha]}{color.split("#")[1]}'

        colors = itertools.cycle(COLORS)

        for parent_no, child_no, _, child, child_depth in self.root.iter_edges():

                if child_depth <= max_depth:

                    if isinstance(child, Branch):

                        text = f'{child.split} \n {child.target_dist} \n samples: {child.n_samples}'

                    elif isinstance(child, Leaf):

                        text = f'{child.target_dist} \n samples: {child.n_samples}'

                    mode = child.target_dist.mode

                    if mode is not None:
                        fillcolor = str(apply_alpha(colors[mode],
                            child.target_dist.pmf(child.target_dist.mode))
                        )
                    else:
                        fillcolor = '#FFFFFF'

                    dot.node(f'{child_no}', text, fillcolor=fillcolor, style='filled')
                    dot.edge(f'{parent_no}', f'{child_no}')
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
        1. `Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams. In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 71-80). <https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf>`_
        2. `Article by The Morning Paper <https://blog.acolyer.org/2015/08/26/mining-high-speed-data-streams/>`_

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

COLORS = [
    '#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5F5', '#FFFF00', '#9ACD32',
    '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887',
    '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF',
    '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#A9A9A9', '#BDB76B', '#8B008B',
    '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F',
    '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#696969', '#1E90FF',
    '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700', '#DAA520',
    '#808080', '#008000', '#ADFF2F', '#808080', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082',
    '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080',
    '#E0FFFF', '#FAFAD2', '#D3D3D3', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA',
    '#87CEFA', '#778899', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6',
    '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE',
    '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD',
    '#000080', '#FDF5E6', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA',
    '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD',
    '#B0E0E6', '#800080', '#663399', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072',
    '#F4A460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090',
    '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347',
    '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF'
]
