from __future__ import annotations

import math

from river import base, stats
from river.tree.base import Branch, Leaf
from river.tree.mondrian._mondrian_ops import (  # type: ignore[import-not-found]
    log_sum_2_exp_c,
    predict_scores_c,
    range_extension_c,
    update_ranges_c,
)


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

    is_leaf = True

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
        return f"{self.feature} <= {self.threshold}"


class MondrianNode(base.Base):
    """Representation of a node within a Mondrian tree"""

    # Flag to distinguish leaves from branches without isinstance checks
    is_leaf: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_range_min: dict = {}
        self.memory_range_max: dict = {}

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

        if self.is_leaf:
            return

        depth += 1

        left, right = self.children
        left.update_depth(depth)
        right.update_depth(depth)

    def update_weight_tree(self):
        """Update the weight of the node in the tree."""

        if self.is_leaf:
            self.log_weight_tree = self.weight
        else:
            left, right = self.children
            self.log_weight_tree = log_sum_2_exp_c(
                self.weight, left.log_weight_tree + right.log_weight_tree
            )

    def range(self, feature) -> tuple[float, float]:
        """Output the known range of the node regarding the given feature.

        Parameters
        ----------
        feature
            Feature for which you want to know the range.

        """

        return (
            self.memory_range_min[feature],
            self.memory_range_max[feature],
        )

    def range_extension(self, x) -> tuple[float, dict]:
        """Compute the range extension of the node for the given sample.

        Parameters
        ----------
        x
            Sample dict.

        """

        return range_extension_c(self.memory_range_min, self.memory_range_max, x)


class MondrianNodeClassifier(MondrianNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_samples = 0
        self.counts: list = []

    def replant(self, leaf: MondrianNodeClassifier, copy_all: bool = False):
        """Transfer information from a leaf to a new branch."""
        self.weight = leaf.weight  # type: ignore
        self.log_weight_tree = leaf.log_weight_tree  # type: ignore

        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def score(self, y_idx: int, dirichlet: float, n_classes: int) -> float:
        """Compute the score of the node.

        Parameters
        ----------
        y_idx
            Class index for which we want the score.
        dirichlet
            Dirichlet parameter of the tree.
        n_classes
            The total number of classes seen so far.

        """

        counts = self.counts
        count = counts[y_idx] if y_idx < len(counts) else 0
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(self, dirichlet: float, n_classes: int) -> list:
        """Predict the scores of all classes and output a list of scores.

        Parameters
        ----------
        dirichlet
            Dirichlet parameter of the tree.
        n_classes
            The total number of classes of the problem.

        """

        return predict_scores_c(self.counts, len(self.counts), n_classes, dirichlet, self.n_samples)

    def loss(self, y_idx: int, dirichlet: float, n_classes: int) -> float:
        """Compute the loss of the node.

        Parameters
        ----------
        y_idx
            Class index.
        dirichlet
            Dirichlet parameter of the problem.
        n_classes
            The total number of classes of the problem.

        """

        sc = self.score(y_idx, dirichlet, n_classes)
        return -math.log(sc)

    def update_weight(
        self,
        y_idx: int,
        dirichlet: float,
        use_aggregation: bool,
        step: float,
        n_classes: int,
    ) -> float:
        """Update the weight of the node given a class index and the method used.

        Parameters
        ----------
        y_idx
            Class index of a given sample.
        dirichlet
            Dirichlet parameter of the tree.
        use_aggregation
            Whether to use aggregation or not during computation (given by the tree).
        step
            Step parameter of the tree.
        n_classes
            The total number of classes of the problem.

        """

        loss_t = self.loss(y_idx, dirichlet, n_classes)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, y_idx):
        """Update the count for the given class index.

        Parameters
        ----------
        y_idx
            Class index of a given sample.

        """

        counts = self.counts
        if y_idx >= len(counts):
            counts.extend([0] * (y_idx + 1 - len(counts)))
        counts[y_idx] += 1

    def is_dirac(self, y_idx: int) -> bool:
        """Check whether the node is pure regarding the given class index.

        Parameters
        ----------
        y_idx
            Class index of a given sample.

        """

        counts = self.counts
        if y_idx >= len(counts):
            return self.n_samples == 0
        return self.n_samples == counts[y_idx]

    def update_downwards(
        self,
        x,
        y_idx: int,
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
            Sample dict.
        y_idx
            Class index of the sample.
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

        # Update ranges
        if self.n_samples == 0:
            self.memory_range_min = dict(x)
            self.memory_range_max = dict(x)
        else:
            update_ranges_c(self.memory_range_min, self.memory_range_max, x)

        # One more sample in the node
        self.n_samples += 1

        # Inline weight update to avoid method call overhead
        if do_update_weight and use_aggregation:
            counts = self.counts
            count = counts[y_idx] if y_idx < len(counts) else 0
            sc = (count + dirichlet) / (self.n_samples + dirichlet * n_classes)
            self.weight += step * math.log(sc)

        # Update count
        counts = self.counts
        if y_idx >= len(counts):
            counts.extend([0] * (y_idx + 1 - len(counts)))
        counts[y_idx] += 1


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

    is_leaf = True

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
        self._mean = stats.Mean()

    def replant(self, leaf: MondrianNodeRegressor, copy_all: bool = False):
        """Transfer information from a leaf to a new branch."""
        self.weight = leaf.weight  # type: ignore
        self.log_weight_tree = leaf.log_weight_tree  # type: ignore
        self._mean = stats.Mean._from_state(leaf._mean.n, leaf._mean.get())

        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def predict(self) -> base.typing.RegTarget:
        """Return the prediction of the node."""
        return self._mean.get()

    def loss(self, sample_value: base.typing.RegTarget) -> float:
        """Compute the loss of the node.

        Parameters
        ----------
        sample_value
            A given value.

        """

        r = self._mean.get() - sample_value  # type: ignore
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
            Sample dict.
        sample_value
            Label of the sample.
        use_aggregation
            Should it use the aggregation or not
        step
            Step of the tree.
        do_update_weight
            Should we update the weights of the node as well.

        """

        # Update ranges
        if self.n_samples == 0:
            self.memory_range_min = dict(x)
            self.memory_range_max = dict(x)
        else:
            update_ranges_c(self.memory_range_min, self.memory_range_max, x)

        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            self.update_weight(sample_value, use_aggregation, step)

        # Update the mean of the labels in the node online
        self._mean.update(sample_value)


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

    is_leaf = True

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
