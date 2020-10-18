import math

from river.proba.base import ContinuousDistribution

from .. import base


class Branch(base.Branch):

    def update(self, x, y):
        if self.split(x):
            self.left = self.left.update(x, y)
            return self
        self.right = self.right.update(x, y)
        return self


class Leaf(base.Leaf):

    def __init__(self, depth, tree, target_dist):
        self.depth = depth
        self.tree = tree
        self.target_dist = target_dist
        self.n_samples = 0
        self.split_enums = {}

    @property
    def n_classes(self):
        """The number of observed classes."""
        if isinstance(self.target_dist, ContinuousDistribution):
            raise AttributeError('The target is continuous, hence there are not classes')
        return len(self.target_dist)

    @property
    def is_pure(self):
        try:
            return self.n_classes < 2
        except AttributeError:
            return self.n_samples > 1

    @property
    def hoeffding_bound(self):
        """Returns the current Hoeffding bound."""
        R = math.log(self.n_classes)
        n = self.n_samples
        δ = self.tree.confidence
        return math.sqrt(R ** 2 * math.log(1 / δ) / (2 * n))

    def update(self, x, y):

        # Update the class counts
        self.target_dist.update(y)
        self.n_samples += 1

        # Update the sufficient statistics of each feature's split searcher
        for i, xi in x.items():
            try:
                ss = self.split_enums[i]
            except KeyError:
                ss = self.split_enums[i] = self.tree._get_split_enum(value=xi)
            ss.update(x=xi, y=y)

        # Check if splitting is authorized or not
        if (
            self.depth >= self.tree.max_depth or
            self.is_pure or
            self.n_samples % self.tree.patience != 0
        ):
            return self

        # Search for the best split given the current information
        top_2_diff, split = self.find_best_split()
        if not split:
            return self

        # Calculate the Hoeffding bound
        ε = self.hoeffding_bound

        if top_2_diff > ε or ε < self.tree.tie_threshold:
            return Branch(
                split=split,
                left=Leaf(depth=self.depth + 1, tree=self.tree, target_dist=self.tree._make_leaf_dist()),
                right=Leaf(depth=self.depth + 1, tree=self.tree, target_dist=self.tree._make_leaf_dist()),
                tree=self.tree,
                target_dist=self.target_dist,
                n_samples=self.n_samples
            )
        return self

    def find_best_split(self):
        """Returns the best potential split."""

        current_impurity = self.tree.criterion_func(dist=self.target_dist)
        best_gain = -math.inf
        second_best_gain = -math.inf
        best_split = None

        # For each feature
        for feature_name, split_enum in self.split_enums.items():

            # For each candidate split
            for how, at, l_dist, r_dist in split_enum.enumerate_splits(target_dist=self.target_dist):

                # Ignore the split if it results in a new leaf with not enough samples
                if (
                    l_dist.n_samples < self.tree.min_child_samples or
                    r_dist.n_samples < self.tree.min_child_samples
                ):
                    continue

                # Compute the decrease in impurity brought by the split
                left_impurity = self.tree.criterion_func(dist=l_dist)
                right_impurity = self.tree.criterion_func(dist=r_dist)
                impurity = l_dist.n_samples * left_impurity + r_dist.n_samples * right_impurity
                impurity /= l_dist.n_samples + r_dist.n_samples
                gain = current_impurity - impurity

                # Ignore the split if the gain in impurity is too low
                if gain < self.tree.min_split_gain:
                    continue

                # Check if the gain brought by the candidate split is better than the current best
                if gain > best_gain:
                    best_gain, second_best_gain = gain, best_gain
                    best_split = base.Split(on=feature_name, how=how, at=at)
                elif gain > second_best_gain:
                    second_best_gain = gain

        return best_gain - second_best_gain, best_split
