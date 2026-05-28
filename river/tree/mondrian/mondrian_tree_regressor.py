from __future__ import annotations

from river import base
from river.stats._rust_stats import (
    go_downwards_regressor as go_downwards_regressor_c,
)
from river.stats._rust_stats import (
    go_upwards as go_upwards_c,
)
from river.stats._rust_stats import (
    predict_one_regressor as predict_one_regressor_c,
)
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import (
    MondrianBranchRegressor,
    MondrianLeafRegressor,
    MondrianNodeRegressor,
)


class MondrianTreeRegressor(MondrianTree, base.Regressor):
    """Mondrian Tree Regressor.

    By default, this implementation assumes that all feature values are scaled between 0 and 1.
    If you cannot assume the minimum and maximum values for each feature,
    you can use preprocessing.MinMaxScaler as an initial preprocessing step.
    This is important because Mondrian trees are highly sensitive to feature scaling, as
    the distance between a sample and the node's bounding box is calculated as the sum of the distances across all features.

    Parameters
    ----------
    step
        Step of the tree.
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    iteration
        Number iterations to do during training.
    max_nodes
        Maximum number of nodes allowed in the tree. No new splits will occur once this
        limit is reached. If `None`, the tree grows without bound.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The Mondrian Tree Regressor is a type of decision tree that bases splitting decisions over a
    Mondrian process.

    References
    ----------
    [^1]: Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh. Mondrian Forests: Efficient Online Random Forests.
        arXiv:1406.2673, pages 2-4.

    """

    def __init__(
        self,
        step: float = 1.0,
        use_aggregation: bool = True,
        iteration: int = 0,
        max_nodes: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            step=step,
            loss="least-squares",
            use_aggregation=use_aggregation,
            iteration=iteration,
            max_nodes=max_nodes,
            seed=seed,
        )
        # Controls the randomness in the tree
        self.seed = seed

        # The current sample being proceeded
        self._x: dict[base.typing.FeatureName, int | float]
        # The current label index being proceeded
        self._y: base.typing.RegTarget

        # Initialization of the root of the tree
        # It's the root so it doesn't have any parent (hence None)
        self._root = MondrianLeafRegressor(None, 0.0, 0)

    def _is_initialized(self):
        """Check if the tree has learnt at least one sample"""
        return self.iteration != 0

    def _predict(self, node: MondrianNodeRegressor) -> base.typing.RegTarget:
        """Compute the prediction.

        Parameters
        ----------
        node
            Node to make predictions.

        """

        return node.predict()  # type: ignore

    def _split(
        self,
        node: MondrianLeafRegressor | MondrianBranchRegressor,
        split_time: float,
        threshold: float,
        feature: base.typing.FeatureName,
        is_right_extension: bool,
    ) -> MondrianBranchRegressor:
        """Split the given node and attributes the split time, threshold, etc... to the node.

        Parameters
        ----------
        node
            Target node.
        split_time
            Split time of the node in the Mondrian process.
        threshold
            Threshold of acceptance of the node.
        feature
            Feature of the node.
        is_right_extension
            Should we extend the tree in the right or left direction.

        """

        new_depth = node.depth + 1

        # To calm down mypy
        left: MondrianLeafRegressor | MondrianBranchRegressor
        right: MondrianLeafRegressor | MondrianBranchRegressor

        # The node is already a branch: we create a new branch above it and move the existing
        # node one level down the tree
        if isinstance(node, MondrianBranchRegressor):
            old_left, old_right = node.children
            if is_right_extension:
                left = MondrianBranchRegressor(
                    node, split_time, new_depth, node.feature, node.threshold
                )
                right = MondrianLeafRegressor(node, split_time, new_depth)
                left.replant(node, True)

                old_left.parent = left
                old_right.parent = left

                left.children = (old_left, old_right)
            else:
                right = MondrianBranchRegressor(
                    node, split_time, new_depth, node.feature, node.threshold
                )
                left = MondrianLeafRegressor(node, split_time, new_depth)
                right.replant(node, True)

                old_left.parent = right
                old_right.parent = right

                right.children = (old_left, old_right)

            # Update the level of the modified nodes
            new_depth += 1
            old_left.update_depth(new_depth)
            old_right.update_depth(new_depth)

            # Update split info
            node.feature = feature
            node.threshold = threshold
            node.children = (left, right)

            return node

        # We promote the leaf to a branch
        branch = MondrianBranchRegressor(node.parent, node.time, node.depth, feature, threshold)
        left = MondrianLeafRegressor(branch, split_time, new_depth)
        right = MondrianLeafRegressor(branch, split_time, new_depth)
        branch.children = (left, right)

        # Copy properties from the previous leaf
        branch.replant(node, True)

        if is_right_extension:
            left.replant(node, True)
        else:
            right.replant(node, True)

        # To avoid leaving garbage behind
        del node

        return branch

    def _go_downwards(self):
        """Update the tree (downward procedure)."""

        rng = self._rng

        leaf, new_root, nodes_added = go_downwards_regressor_c(
            self._root,
            self._x,
            self._y,
            self.use_aggregation,
            self.step,
            self.iteration,
            self.max_nodes if self.max_nodes is not None else -1,
            self._n_nodes,
            rng.random,
            rng.choices,
            rng.uniform,
            self._split,
        )
        self._n_nodes += nodes_added
        if new_root is not None:
            self._root = new_root
        return leaf

    def _go_upwards(self, leaf: MondrianLeafRegressor):
        """Update the tree (upwards procedure)."""
        go_upwards_c(leaf, self.iteration)

    def learn_one(self, x, y):
        # Setting current sample. Copy x so the caller can safely mutate the
        # input dict after learn_one without disturbing the tree's stored
        # bounding boxes.
        self._x = dict(x)
        self._y = y

        # Learning step
        leaf = self._go_downwards()
        if self.use_aggregation:
            self._go_upwards(leaf)

        # Incrementing iteration
        self.iteration += 1

    def predict_one(self, x):
        """Predict the label of the samples.

        Parameters
        ----------
        x
            Feature vector.

        """

        if not self._is_initialized:
            return

        return predict_one_regressor_c(self._root, x, self.use_aggregation)
