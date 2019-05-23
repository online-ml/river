import bisect
import collections
import functools
import math
import operator

try:
    import graphviz
    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False

from .. import base
from .. import utils

from . import enum


class Branch:

    def __init__(self, feature, values, left, right, tree):
        self.feature = feature
        self.values = set(values)
        self.left = left
        self.right = right
        self.tree = tree

    def go_left(self, x):
        value = x[self.feature]

        if isinstance(value, float):
            value = bisect.bisect_left(self.tree.histograms[self.feature].sorted_bins, value)

        return value in self.values

    def get_leaf(self, x):
        if self.go_left(x):
            return self.left.get_leaf(x)
        return self.right.get_leaf(x)

    def update(self, x, y):
        if self.go_left(x):
            self.left = self.left.update(x, y)
        else:
            self.right = self.right.update(x, y)
        return self


class Leaf:

    def __init__(self, depth, tree):
        self.depth = depth
        self.tree = tree
        self.class_counts = collections.Counter()
        self.n_samples = 0
        dd = collections.defaultdict
        self.feature_counts = dd(functools.partial(dd, collections.Counter))

    def get_leaf(self, x):
        return self

    def update(self, x, y):

        # Update the leaf's overall class counts
        self.class_counts.update((y,))
        self.n_samples += 1

        # Update the leaf's class counts per feature value
        for feature, value in x.items():

            # Continuous values are discretized
            if isinstance(value, float):
                # Update the feature's histogram
                self.tree.histograms[feature].update(value)
                # Discretize the value using the histogram
                value = bisect.bisect_left(self.tree.histograms[feature].sorted_bins, value)

            self.feature_counts[feature][value].update((y,))

        # Check if it worth search for a potential split or not
        if self.depth >= self.tree.max_depth or \
           self.n_samples < self.tree.min_samples_split or \
           (self.tree.min_samples_split + self.n_samples) % self.tree.patience != 0 or \
           self.is_pure:
            return self

        # Search for the best split given the current information
        best_gain, second_best_gain, best_feature, best_values = find_best_split(
            class_counts=self.class_counts,
            feature_counts=self.feature_counts,
            split_enum=self.tree.split_enum
        )

        # Calculate the Hoeffding bound
        R = math.log(len(self.class_counts))
        n = self.n_samples
        δ = self.tree.delta
        bound = math.sqrt(R ** 2 * math.log(1 / δ) / (2 * n))

        diff = best_gain - second_best_gain
        if diff > bound or diff < self.tree.bound_threshold:
            return Branch(
                feature=best_feature,
                values=best_values,
                left=Leaf(depth=self.depth + 1, tree=self.tree),
                right=Leaf(depth=self.depth + 1, tree=self.tree),
                tree=self.tree
            )
        return self

    @property
    def is_pure(self):
        return len(self.class_counts) <= 1


def sum_counters(counters):
    return functools.reduce(operator.add, counters, collections.Counter())


def find_best_split(class_counts, feature_counts, split_enum):
    """

    >>> class_counts = {'slow': 2, 'fast': 2}
    >>> feature_counts = {
    ...     'grade': {
    ...         'steep': collections.Counter({'slow': 2, 'fast': 1}),
    ...         'flat': collections.Counter({'fast': 1})
    ...     },
    ...     'bumpiness': {
    ...         'bumpy': collections.Counter({'slow': 1, 'fast': 1}),
    ...         'smooth': collections.Counter({'slow': 1, 'fast': 1})
    ...     },
    ...     'speed_limit': {
    ...         'yes': collections.Counter({'slow': 2}),
    ...         'no': collections.Counter({'fast': 2})
    ...     }
    ... }
    >>> split_enum = enum.UnaryEnumerator()

    >>> find_best_split(class_counts, feature_counts, split_enum)
    (1.0, 0.311278..., 'speed_limit', ['no'])

    """

    best_gain = -math.inf
    second_best_gain = -math.inf
    best_feature = None
    best_values = None

    current_entropy = utils.entropy(class_counts)

    for feature, counts in feature_counts.items():

        for left, right in split_enum(sorted(counts.keys())):

            left_counts = sum_counters(counts[v] for v in left)
            right_counts = sum_counters(counts[v] for v in right)
            left_total = sum(left_counts.values())
            right_total = sum(right_counts.values())

            entropy = left_total * utils.entropy(left_counts) + \
                right_total * utils.entropy(right_counts)
            entropy /= (left_total + right_total)

            gain = current_entropy - entropy

            if gain > best_gain:
                best_gain, second_best_gain = gain, best_gain
                best_feature = feature
                best_values = left
            elif gain > second_best_gain and gain != best_gain:
                second_best_gain = gain

    return best_gain, second_best_gain, best_feature, best_values


class DecisionTreeClassifier(base.MultiClassifier):
    """

    Parameters:
        max_bins (int): Maximum number of bins used for discretizing continuous values when using
            `utils.Histogram`.

    Attributes:
        histograms (collections.defaultdict)

    """

    def __init__(self, max_depth=5, min_samples_split=10, patience=10, max_bins=30):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.patience = patience
        self.max_bins = max_bins
        self.split_enum = enum.ContiguousEnumerator()  # enum.UnaryEnumerator()
        self.delta = 0.1
        self.bound_threshold = 0.1
        self.histograms = collections.defaultdict(functools.partial(
            utils.Histogram,
            max_bins=max_bins
        ))
        self.root = Leaf(depth=0, tree=self)

    def fit_one(self, x, y):
        self.root = self.root.update(x, y)
        return self

    def predict_proba_one(self, x):
        leaf = self.root.get_leaf(x)
        return {
            label: count / leaf.n_samples
            for label, count in leaf.class_counts.items()
        }

    def to_dot(self):
        """Returns a GraphViz representation of the decision tree."""

        if not GRAPHVIZ_INSTALLED:
            raise RuntimeError('graphviz is not installed')

        dot = graphviz.Digraph()

        def add_node(node, code):

            # Draw the current node
            if isinstance(node, Leaf):
                dot.node(code, str(node.class_counts))
            else:
                dot.node(code, f'{node.feature}\n{str(node.values)}')
                add_node(node.left, f'{code}0')
                add_node(node.right, f'{code}1')

            # Draw the link with the previous node
            is_root = len(code) == 1
            if not is_root:
                dot.edge(code[:-1], code)

        add_node(self.root, '0')

        return dot
