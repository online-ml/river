import collections
import functools
import math

from . import branch
from . import splitting


class Leaf:
    """

    ``feature_counts`` is a table that counts the occurrences of each value for each feature and
    each class. These counts are stored as a nested dictionary. The first level is for the
    feature names. The second level contains the values of each feature. The third level contains
    the class counts. Here is an example:

        feature_counts = {
            'color': {
                'black': collections.Counter({False: 64, True: 15}),
                'yellow': collections.Counter({True: 86})
            },
            'age': {
                0: collections.Counter({False: 9, True: 33}),
                1: collections.Counter({False: 15, True: 36})
                2: collections.Counter({False: 40, True: 2})
            }
        }

    In our implementation, continuous values are always discretized using an online histogram. This
    explains why in the above example the possible `age` values are 0, 1, 2. These values represent
    the bin in which each age belongs. There are thus 3 age groups in the above example. 33
    observations are part of the first age group.

    """

    __slots__ = 'depth', 'tree', 'class_counts', 'n_samples', 'feature_counts'

    def __init__(self, depth, tree, class_counts=None, feature_counts=None):
        self.depth = depth
        self.tree = tree
        self.class_counts = collections.Counter({} if class_counts is None else class_counts)
        self.n_samples = 0
        dd = collections.defaultdict
        self.feature_counts = dd(functools.partial(dd, collections.Counter))

        # Add initial feature counts if there are any
        if feature_counts is not None:
            for feature, counts in feature_counts.items():
                for value, value_class_counts in counts.items():
                    self.feature_counts[feature][value].update(value_class_counts)

    def get_leaf(self, x):
        return self

    def update(self, x, y):

        # Update the leaf's overall class counts
        self.class_counts.update((y,))
        self.n_samples += 1

        # Update the leaf's class counts for each feature value
        for feature, value in x.items():

            # Continuous values are discretized
            if isinstance(value, float):
                # Update the feature's histogram
                self.tree.histograms[feature].update(value)
                # Discretize the value using the histogram
                value = self.tree.histograms[feature].bin(value)

            self.feature_counts[feature][value].update((y,))

        # Check if it is worth searching for a potential split or not
        if self.depth >= self.tree.max_depth or \
           self.n_samples < self.tree.min_samples_split or \
           (self.tree.min_samples_split + self.n_samples) % self.tree.patience != 0 or \
           self.is_pure:
            return self

        # Search for the best split given the current information
        top_2_diff, split = splitting.search_split_info_gain(
            class_counts=self.class_counts,
            feature_counts=self.feature_counts,
            categoricals=set(self.feature_counts.keys()) - set(self.tree.histograms.keys())
        )

        # Calculate the Hoeffding bound
        R = math.log(len(self.class_counts))
        n = self.n_samples
        δ = self.tree.delta
        ε = math.sqrt(R ** 2 * math.log(1 / δ) / (2 * n))  # Hoeffding bound

        if top_2_diff > ε or ε < self.tree.bound_threshold:

            if split.feature in self.tree.histograms:
                split.value = self.tree.histograms[feature].sorted_bins[split.value + 1]

            return branch.Branch(
                split=split,
                left=Leaf(depth=self.depth + 1, tree=self.tree),
                right=Leaf(depth=self.depth + 1, tree=self.tree),
                tree=self.tree
            )
        return self

    @property
    def is_pure(self):
        return len(self.class_counts) <= 1
