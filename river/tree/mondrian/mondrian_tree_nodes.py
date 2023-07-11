from __future__ import annotations

import collections
import math

from river import base, stats
from river.tree.base import Branch, Leaf
from river.utils.math import log_sum_2_exp


class MondrianLeaf(Leaf):
    """Prototype class for all types of nodes in a Mondrian Tree.

    Parameters
    ----------
    parent
        Parent Node.
    time
        Split time of the node for Mondrian process.
    depth
        Depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        super().__init__()

        # Generic Node attributes
        self.parent = parent
        self.time = time
        self.depth = depth

    @property
    def __repr__(self):
        return f"MondrianLeaf : {self.parent}, {self.time}, {self.depth}"


class MondrianBranch(Branch):
    def __init__(self, parent, time, depth, feature, threshold, *children):
        super().__init__(*children)

        self.parent = parent
        self.time = time
        self.depth = depth
        self.feature = feature
        self.threshold = threshold

    def branch_no(self, x) -> int:
        if x[self.feature] <= self.threshold:
            return 0
        return 1

    def next(self, x):
        return self.children[self.branch_no(x)]

    def most_common_path(self):
        left, right = self.children

        if left.weight < right.weight:  # type: ignore
            return 1, right
        return 0, left

    def repr_split(self):
        return f"{self.feature} ≤ {self.threshold}"


class MondrianNode(base.Base):
    """Representation of a node within a Mondrian tree"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_range_min = collections.defaultdict(int)
        self.memory_range_max = collections.defaultdict(int)

        self.weight = 0.0
        self.log_weight_tree = 0.0

    def update_depth(self, depth):
        """Update the depth of the current node with the given depth.

        Parameters
        ----------
        depth
            Depth of the node.

        """

        self.depth = depth

        if isinstance(self, MondrianLeaf):
            return

        depth += 1

        left, right = self.children
        left.update_depth(depth)
        right.update_depth(depth)

    def update_weight_tree(self):
        """Update the weight of the node in the tree."""

        if isinstance(self, MondrianLeaf):
            self.log_weight_tree = self.weight
        else:
            left, right = self.children
            self.log_weight_tree = log_sum_2_exp(
                self.weight, left.log_weight_tree + right.log_weight_tree
            )

    def range(self, feature) -> tuple[float, float]:
        """Output the known range of the node regarding the j-th feature.

        Parameters
        ----------
        feature
            Feature for which you want to know the range.

        """

        return (
            self.memory_range_min[feature],
            self.memory_range_max[feature],
        )

    def range_extension(self, x) -> tuple[float, dict[base.typing.ClfTarget, float]]:
        """Compute the range extension of the node for the given sample.

        Parameters
        ----------
        x
            Sample to deal with.

        """

        extensions: dict[base.typing.ClfTarget, float] = {}
        extensions_sum = 0.0
        for feature in x:
            x_f = x[feature]
            feature_min_j, feature_max_j = self.range(feature)
            if x_f < feature_min_j:
                diff = feature_min_j - x_f
            elif x_f > feature_max_j:
                diff = x_f - feature_max_j
            else:
                diff = 0
            extensions[feature] = diff
            extensions_sum += diff
        return extensions_sum, extensions


class MondrianNodeClassifier(MondrianNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_samples = 0
        self.counts = collections.defaultdict(int)

    def replant(self, leaf: MondrianNodeClassifier, copy_all: bool = False):
        """Transfer information from a leaf to a new branch."""
        self.weight = leaf.weight  # type: ignore
        self.log_weight_tree = leaf.log_weight_tree  # type: ignore

        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def score(self, y: base.typing.ClfTarget, dirichlet: float, n_classes: int) -> float:
        """Compute the score of the node.

        Parameters
        ----------
        y
            Class for which we want the score.
        dirichlet
            Dirichlet parameter of the tree.
        n_classes
            The total number of classes seen so far.

        Notes
        -----
        This uses Jeffreys prior with Dirichlet parameter for smoothing.

        """

        count = self.counts[y]

        # We use the Jeffreys prior with dirichlet parameter
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(
        self, dirichlet: float, classes: set, n_classes: int
    ) -> dict[base.typing.ClfTarget, float]:
        """Predict the scores of all classes and output a `scores` dictionary
        with the new values.

        Parameters
        ----------
        dirichlet
            Dirichlet parameter of the tree.
        classes
            The set of classes seen so far
        n_classes
            The total number of classes of the problem.

        """

        scores = {}
        for c in classes:
            scores[c] = self.score(c, dirichlet, n_classes)
        return scores

    def loss(self, y: base.typing.ClfTarget, dirichlet: float, n_classes: int) -> float:
        """Compute the loss of the node.

        Parameters
        ----------
        y
            A given class of the problem.
        dirichlet
            Dirichlet parameter of the problem.
        n_classes
            The total number of classes of the problem.

        """

        sc = self.score(y, dirichlet, n_classes)
        return -math.log(sc)

    def update_weight(
        self,
        y: base.typing.ClfTarget,
        dirichlet: float,
        use_aggregation: bool,
        step: float,
        n_classes: int,
    ) -> float:
        """Update the weight of the node given a class and the method used.

        Parameters
        ----------
        y
            Class of a given sample.
        dirichlet
            Dirichlet parameter of the tree.
        use_aggregation
            Whether to use aggregation or not during computation (given by the tree).
        step
            Step parameter of the tree.
        n_classes
            The total number of classes of the problem.

        """

        loss_t = self.loss(y, dirichlet, n_classes)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, y):
        """Update the amount of samples that belong to a class in the node
        (not to use twice if you add one sample).

        Parameters
        ----------
        y
            Class of a given sample.

        """

        self.counts[y] += 1

    def is_dirac(self, y: base.typing.ClfTarget) -> bool:
        """Check whether the node follows a dirac distribution regarding the given
        class, i.e., if the node is pure regarding the given class.

        Parameters
        ----------
        y
            Class of a given sample.

        """

        return self.n_samples == self.counts[y]

    def update_downwards(
        self,
        x,
        y: base.typing.ClfTarget,
        dirichlet: float,
        use_aggregation: bool,
        step: float,
        do_update_weight: bool,
        n_classes: int,
    ):
        """Update the node when running a downward procedure updating the tree.

        Parameters
        ----------
        x
            Sample to proceed.
        y
            Class of the sample x.
        dirichlet
            Dirichlet parameter of the tree.
        use_aggregation
            Should it use the aggregation or not
        step
            Step of the tree.
        do_update_weight
            Should we update the weights of the node as well.
        n_classes
            The total number of classes of the problem.

        """

        # Updating the range of the feature values known by the node
        # If it is the first sample, we copy the features vector into the min and max range
        if self.n_samples == 0:
            for feature in x:
                x_f = x[feature]
                self.memory_range_min[feature] = x_f
                self.memory_range_max[feature] = x_f
        # Otherwise, we update the range
        else:
            for feature in x:
                x_f = x[feature]
                if x_f < self.memory_range_min[feature]:
                    self.memory_range_min[feature] = x_f
                if x_f > self.memory_range_max[feature]:
                    self.memory_range_max[feature] = x_f

        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            self.update_weight(y, dirichlet, use_aggregation, step, n_classes)

        self.update_count(y)


class MondrianLeafClassifier(MondrianNodeClassifier, MondrianLeaf):
    """Mondrian Tree Classifier leaf node.

    Parameters
    ----------
    parent
        Parent node.
    time
        Split time of the node.
    depth
        The depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        super().__init__(parent, time, depth)


class MondrianBranchClassifier(MondrianNodeClassifier, MondrianBranch):
    """Mondrian Tree Classifier branch node.

    Parameters
    ----------
    parent
        Parent node of the branch.
    time
        Split time characterizing the branch.
    depth
        Depth of the branch in the tree.
    feature
        Feature of the branch.
    threshold
        Acceptation threshold of the branch.
    *children
        Children nodes of the branch.

    """

    def __init__(self, parent, time, depth, feature, threshold, *children):
        super().__init__(parent, time, depth, feature, threshold, *children)


class MondrianNodeRegressor(MondrianNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_samples = 0
        self.mean = stats.Mean()

    def replant(self, leaf: MondrianNodeRegressor, copy_all: bool = False):
        """Transfer information from a leaf to a new branch."""
        self.weight = leaf.weight  # type: ignore
        self.log_weight_tree = leaf.log_weight_tree  # type: ignore
        self.mean = leaf.mean

        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def predict(self) -> base.typing.RegTarget:
        """Return the prediction of the node."""
        return self.mean.get()

    def loss(self, sample_value: base.typing.RegTarget) -> float:
        """Compute the loss of the node.

        Parameters
        ----------
        sample_value
            A given value.

        """

        r = self.predict() - sample_value  # type: ignore
        return r * r / 2

    def update_weight(
        self,
        sample_value: base.typing.RegTarget,
        use_aggregation: bool,
        step: float,
    ) -> float:
        """Update the weight of the node given a label and the method used.

        Parameters
        ----------
        sample_value
            Label of a given sample.
        use_aggregation
            Whether to use aggregation or not during computation (given by the tree).
        step
            Step parameter of the tree.

        """

        loss_t = self.loss(sample_value)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_downwards(
        self,
        x,
        sample_value: base.typing.RegTarget,
        use_aggregation: bool,
        step: float,
        do_update_weight: bool,
    ):
        """Update the node when running a downward procedure updating the tree.

        Parameters
        ----------
        x
            Sample to proceed (as a list).
        sample_value
            Label of the sample x.
        use_aggregation
            Should it use the aggregation or not
        step
            Step of the tree.
        do_update_weight
            Should we update the weights of the node as well.

        """

        # Updating the range of the feature values known by the node
        # If it is the first sample, we copy the features vector into the min and max range
        if self.n_samples == 0:
            for feature in x:
                x_f = x[feature]
                self.memory_range_min[feature] = x_f
                self.memory_range_max[feature] = x_f
        # Otherwise, we update the range
        else:
            for feature in x:
                x_f = x[feature]
                if x_f < self.memory_range_min[feature]:
                    self.memory_range_min[feature] = x_f
                if x_f > self.memory_range_max[feature]:
                    self.memory_range_max[feature] = x_f

        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            self.update_weight(sample_value, use_aggregation, step)

        # Update the mean of the labels in the node online
        self.mean.update(sample_value)


class MondrianLeafRegressor(MondrianNodeRegressor, MondrianLeaf):
    """Mondrian Tree Regressor leaf node.

    Parameters
    ----------
    parent
        Parent node.
    time
        Split time of the node.
    depth
        The depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        super().__init__(parent, time, depth)


class MondrianBranchRegressor(MondrianNodeRegressor, MondrianBranch):
    """Mondrian Tree Regressor branch node.

    Parameters
    ----------
    parent
        Parent node of the branch.
    time
        Split time characterizing the branch.
    depth
        Depth of the branch in the tree.
    feature
        Feature of the branch.
    threshold
        Acceptation threshold of the branch.
    *children
        Children nodes of the branch.

    """

    def __init__(self, parent, time, depth, feature, threshold, *children):
        super().__init__(parent, time, depth, feature, threshold, *children)
