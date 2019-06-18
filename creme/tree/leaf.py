import collections
import math
import numbers

from ..proba.base import ContinuousDistribution

from . import branch
from . import splitting


class Leaf:

    __slots__ = 'depth', 'tree', 'class_counts', 'n_samples', 'split_searchers'

    def __init__(self, depth, tree, class_counts=None):
        self.depth = depth
        self.tree = tree
        self.class_counts = collections.Counter(class_counts)
        self.n_samples = 0
        self.split_searchers = {}

    @property
    def size(self):
        return 1

    @property
    def n_classes(self):
        """The number of observed classes."""
        return len(self.class_counts)

    @property
    def is_pure(self):
        return self.n_classes < 2

    def get_leaf(self, x):
        return self

    def update(self, x, y):

        # Update the class counts
        self.class_counts[y] += 1
        self.n_samples += 1

        # Update the sufficient statistics of each feature's split searcher
        for i, xi in x.items():
            try:
                ss = self.split_searchers[i]
            except KeyError:
                ss = self.split_searchers[i] = (
                    splitting.GaussianSplitSearcher(n=30)
                    if isinstance(xi, numbers.Number) else
                    splitting.CategoricalSplitSearcher()
                )
            ss.update(xi, y)

        # Check if splitting is authorized or not
        if (
            self.depth >= self.tree.max_depth or
            self.is_pure or
            self.n_samples % self.tree.patience != 0
        ):
            return self

        # Search for the best split given the current information
        top_2_diff, split, left_class_counts, right_class_counts = self.find_best_split()

        # Calculate the Hoeffding bound
        R = math.log(self.n_classes)
        n = self.n_samples
        δ = self.tree.confidence
        ε = math.sqrt(R ** 2 * math.log2(1 / δ) / (2 * n))  # Hoeffding bound

        if top_2_diff > ε or ε < self.tree.tie_threshold:
            return branch.Branch(
                split=split,
                left=Leaf(depth=self.depth + 1, tree=self.tree),
                right=Leaf(depth=self.depth + 1, tree=self.tree),
                tree=self.tree
            )
        return self

    def find_best_split(self):
        """Returns the best potential split."""

        current_impurity = self.tree.criterion(self.class_counts)
        best_gain = -math.inf
        second_best_gain = -math.inf
        best_split = None
        best_l_class_counts = None
        best_r_class_counts = None

        # For each feature
        for i, ss in self.split_searchers.items():

            # For each candidate split
            for at, op in ss.enumerate_splits():

                # Get the left and right class counts induced the split
                l_class_counts, r_class_counts = ss.do_split(at, self.class_counts)
                l_total = sum(l_class_counts.values())
                r_total = sum(r_class_counts.values())

                # Ignore the split if it results in a new leaf with not enough samples
                if l_total < self.tree.min_child_samples or r_total < self.tree.min_child_samples:
                    continue

                # Compute the gain brought by the split
                l_impurity = self.tree.criterion(l_class_counts)
                r_impurity = self.tree.criterion(r_class_counts)
                impurity = (l_total * l_impurity + r_total * r_impurity) / (l_total + r_total)
                gain = current_impurity - impurity

                # Check if the gain brought by the candidate split is better than the current best
                if gain > best_gain:
                    best_gain, second_best_gain = gain, best_gain
                    best_split = branch.Split(on=i, how=op, at=at)
                    best_l_class_counts = l_class_counts
                    best_r_class_counts = r_class_counts
                elif gain > second_best_gain:
                    second_best_gain = gain

        return best_gain - second_best_gain, best_split, best_l_class_counts, best_r_class_counts

    def predict(self, x):
        total = sum(self.class_counts.values())
        return {label: count / total for label, count in self.class_counts.items()}

    def predict_naive_bayes(self, x):
        """

        Example:

            >>> import itertools
            >>> from creme.tree.splitting import CategoricalSplitSearcher

            >>> leaf = Leaf(0, None)

            >>> counts = [
            ...     ('A1', 'C1', 'A', 12),
            ...     ('A1', 'C1', 'B', 28),
            ...     ('A1', 'C2', 'A', 34),
            ...     ('A1', 'C2', 'B', 26),
            ...     ('A2', 'C1', 'C', 5),
            ...     ('A2', 'C1', 'D', 10),
            ...     ('A2', 'C1', 'E', 25),
            ...     ('A2', 'C2', 'C', 21),
            ...     ('A2', 'C2', 'D', 8),
            ...     ('A2', 'C2', 'E', 31),
            ...     ('A3', 'C1', 'F', 13),
            ...     ('A3', 'C1', 'G', 9),
            ...     ('A3', 'C1', 'H', 3),
            ...     ('A3', 'C1', 'I', 15),
            ...     ('A3', 'C2', 'F', 11),
            ...     ('A3', 'C2', 'G', 21),
            ...     ('A3', 'C2', 'H', 19),
            ...     ('A3', 'C2', 'I', 9)
            ... ]

            >>> for feature, feature_counts in itertools.groupby(counts, key=lambda x: x[0]):
            ...     leaf.split_searchers[feature] = CategoricalSplitSearcher()
            ...     for _, y, x, n in feature_counts:
            ...         for _ in range(n):
            ...             _ = leaf.split_searchers[feature].update(x, y)

            >>> leaf.class_counts = {'C1': 40, 'C2': 60}

            >>> x = {'A1': 'B', 'A2': 'E', 'A3': 'I'}
            >>> leaf.predict(x)
            {'C1': 0.4, 'C2': 0.6}
            >>> leaf.predict_naive_bayes(x)
            {'C1': 0.7650830661614689, 'C2': 0.23491693383853113}

        """
        y_pred = self.predict(x)

        for i, xi in x.items():
            if i in self.split_searchers:
                for label, dist in self.split_searchers[i].items():
                    if isinstance(dist, ContinuousDistribution):
                        y_pred[label] *= dist.pdf(xi)
                    else:
                        y_pred[label] *= dist.pmf(xi)

        total = sum(y_pred.values())

        if total == 0:
            return {label: 1. / len(y_pred) for label in y_pred}

        for label, proba in y_pred.items():
            y_pred[label] /= total

        return y_pred
