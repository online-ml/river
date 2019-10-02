import math

from ..proba.base import ContinuousDistribution


class Branch:

    def __init__(self, split, left, right, tree):
        self.split = split
        self.left = left
        self.right = right
        self.tree = tree

    @property
    def size(self):
        return self.left.size + self.right.size

    def get_leaf(self, x):
        if self.split(x):
            return self.left.get_leaf(x)
        return self.right.get_leaf(x)

    def update(self, x, y):
        if self.split(x):
            self.left = self.left.update(x, y)
            return self
        self.right = self.right.update(x, y)
        return self


class Leaf:

    def __init__(self, depth, tree, target_dist):
        self.depth = depth
        self.tree = tree
        self.target_dist = target_dist
        self.n_samples = 0
        self.split_enums = {}

    @property
    def size(self):
        return 1

    @property
    def n_classes(self):
        """The number of observed classes."""
        if isinstance(self.target_dist, ContinuousDistribution):
            raise ValueError('The target is continuous, hence there are not classes')
        return len(self.target_dist)

    @property
    def is_pure(self):
        try:
            return self.n_classes < 2
        except ValueError:
            return False

    def get_leaf(self, x):
        return self

    @property
    def hoeffding_bound(self):
        """Returns the current Hoeffding bound.

        TODO: handle continuous target
        """
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
                ss = self.split_enums[i] = self.tree._get_split_enum(name=i, value=xi)
            ss.update(x=xi, y=y)

        # Check if splitting is authorized or not
        if (
            self.depth >= self.tree.max_depth or
            self.is_pure or
            self.n_samples % self.tree.patience != 0
        ):
            return self

        # Search for the best split given the current information
        top_2_diff, split, left_dist, right_dist = self.find_best_split()

        # Calculate the Hoeffding bound
        ε = self.hoeffding_bound

        if top_2_diff > ε or ε < self.tree.tie_threshold:
            return Branch(
                split=split,
                left=Leaf(depth=self.depth + 1, tree=self.tree, target_dist=left_dist),
                right=Leaf(depth=self.depth + 1, tree=self.tree, target_dist=right_dist),
                tree=self.tree
            )
        return self

    def find_best_split(self):
        """Returns the best potential split."""

        current_impurity = self.tree.criterion(dist=self.target_dist)
        best_gain = -math.inf
        second_best_gain = -math.inf
        best_split = None
        best_l_dist = None
        best_r_dist = None

        # For each feature
        for ss in self.split_enums.values():

            # For each candidate split
            for split, l_dist, r_dist in ss.enumerate_splits(target_dist=self.target_dist):

                # Ignore the split if it results in a new leaf with not enough samples
                if (
                    l_dist.n_samples < self.tree.min_child_samples or
                    r_dist.n_samples < self.tree.min_child_samples
                ):
                    continue

                # Compute the gain brought by the split
                left_impurity = self.tree.criterion(dist=l_dist)
                right_impurity = self.tree.criterion(dist=r_dist)
                impurity = l_dist.n_samples * left_impurity + r_dist.n_samples * right_impurity
                impurity /= l_dist.n_samples + r_dist.n_samples
                gain = current_impurity - impurity

                # Ignore the split if the gain in impurity is too low
                if gain < self.tree.min_split_gain:
                    continue

                # Check if the gain brought by the candidate split is better than the current best
                if gain > best_gain:
                    best_gain, second_best_gain = gain, best_gain
                    best_split = split
                    best_l_dist = l_dist
                    best_r_dist = r_dist
                elif gain > second_best_gain:
                    second_best_gain = gain

        if best_split is None:
            raise RuntimeError('No best split was found')

        return best_gain - second_best_gain, best_split, best_l_dist, best_r_dist

    def predict(self, x):
        if isinstance(self.target_dist, ContinuousDistribution):
            return self.target_dist.mode
        return {c: self.target_dist.pmf(c) for c in self.target_dist}

    def predict_naive_bayes(self, x):

        y_pred = self.predict(x)

        for i, xi in x.items():
            if i in self.split_enums:
                for label, dist in self.split_enums[i].items():
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
