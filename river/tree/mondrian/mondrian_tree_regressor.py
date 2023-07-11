from __future__ import annotations

import math

from river import base, utils
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import (
    MondrianBranchRegressor,
    MondrianLeafRegressor,
    MondrianNodeRegressor,
)


class MondrianTreeRegressor(MondrianTree, base.Regressor):
    """Mondrian Tree Regressor.

    Parameters
    ----------
    step
        Step of the tree.
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    iteration
        Number iterations to do during training.
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
        step: float = 0.1,
        use_aggregation: bool = True,
        iteration: int = 0,
        seed: int = None,
    ):
        super().__init__(
            step=step,
            loss="least-squares",
            use_aggregation=use_aggregation,
            iteration=iteration,
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

    def _loss(self, node: MondrianNodeRegressor) -> float:
        """Compute the loss for the given node regarding the current label.

        Parameters
        ----------
        node
            Node to evaluate the loss.

        """

        return node.loss(self._y)

    def _update_weight(self, node: MondrianNodeRegressor) -> float:
        """Update the weight of the node regarding the current label with the tree parameters.

        Parameters
        ----------
        node
            Node to update the weight.

        """

        return node.update_weight(self._y, self.use_aggregation, self.step)

    def _update_downwards(self, node: MondrianNodeRegressor, do_update_weight):
        """Update the node when running a downward procedure updating the tree.

        Parameters
        ----------
        node
            Target node.
        do_update_weight
            Whether we should update the weights or not.

        """

        return node.update_downwards(
            self._x, self._y, self.use_aggregation, self.step, do_update_weight
        )

    def _compute_split_time(
        self,
        node: MondrianLeafRegressor | MondrianBranchRegressor,
        extensions_sum: float,
    ) -> float:
        """Computes the split time of the given node.

        Parameters
        ----------
        node
            Target node.

        """

        if extensions_sum > 0:
            # Sample an exponential with intensity = extensions_sum
            T = utils.random.exponential(1 / extensions_sum, rng=self._rng)

            time = node.time
            # Splitting time of the node (if splitting occurs)
            split_time = time + T
            # If the node is a leaf we must split it
            if isinstance(node, MondrianLeafRegressor):
                return split_time
            # Otherwise we apply Mondrian process dark magic :)
            # 1. We get the creation time of the childs (left and right is the same)
            left, _ = node.children
            child_time = left.time
            # 2. We check if splitting time occurs before child creation time
            if split_time < child_time:
                return split_time

        return 0.0

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
            Feature index of the node.
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
                left.replant(node)

                old_left.parent = left
                old_right.parent = left

                left.children = (old_left, old_right)
            else:
                right = MondrianBranchRegressor(
                    node, split_time, new_depth, node.feature, node.threshold
                )
                left = MondrianLeafRegressor(node, split_time, new_depth)
                right.replant(node)

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
            left.replant(node)
        else:
            right.replant(node)

        # To avoid leaving garbage behind
        del node

        return branch

    def _go_downwards(self):
        """Update the tree (downward procedure)."""

        # We update the nodes along the path which leads to the leaf containing the current
        # sample. For each node on the path, we consider the possibility of splitting it,
        # following the Mondrian process definition.

        # We start at the root
        current_node = self._root

        if self.iteration == 0:
            # If it's the first iteration, we just put the current sample in the range of root
            self._update_downwards(current_node, False)
            return current_node
        else:
            # Path from the parent to the current node
            branch_no = None
            while True:
                # Computing the extensions to get the intensities
                extensions_sum, extensions = current_node.range_extension(self._x)

                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = self._compute_split_time(current_node, extensions_sum)

                if split_time > 0:
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path

                    # We normalize the range extensions to get probabilities
                    intensities = utils.norm.normalize_values_in_dict(extensions, inplace=False)

                    # Sample the feature at random with a probability
                    # proportional to the range extensions

                    candidates = sorted(list(self._x.keys()))
                    feature = self._rng.choices(
                        candidates, [intensities[c] for c in candidates], k=1
                    )[0]

                    x_f = self._x[feature]

                    # Is it a right extension of the node ?
                    range_min, range_max = current_node.range(feature)
                    is_right_extension = x_f > range_max
                    if is_right_extension:
                        threshold = self._rng.uniform(range_max, x_f)
                    else:
                        threshold = self._rng.uniform(x_f, range_min)

                    was_leaf = isinstance(current_node, MondrianLeafRegressor)

                    # We split the current node
                    current_node = self._split(
                        current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    # The root node has become a branch
                    if current_node.parent is None:
                        self._root = current_node
                    # Update path from the previous parent to the recently updated node
                    elif was_leaf:
                        parent = current_node.parent
                        if branch_no == 0:
                            parent.children = (current_node, parent.children[1])
                        else:
                            parent.children = (parent.children[0], current_node)

                    # Update the current node
                    self._update_downwards(current_node, True)

                    left, right = current_node.children

                    # Now, get the next node
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    leaf = current_node
                    self._update_downwards(leaf, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to the next one
                    self._update_downwards(current_node, True)
                    if isinstance(current_node, MondrianLeafRegressor):
                        return current_node
                    else:
                        # Save the path direction to keep the tree consistent
                        try:
                            branch_no = current_node.branch_no(self._x)
                            current_node = current_node.children[branch_no]
                        except KeyError:  # Missing split feature
                            branch_no, current_node = current_node.most_common_path()

    def _go_upwards(self, leaf: MondrianLeafRegressor):
        """Update the tree (upwards procedure).

        Parameters
        ----------
        leaf
            Leaf to start from when going upward.

        """

        current_node = leaf

        if self.iteration >= 1:
            while True:
                current_node.update_weight_tree()
                if current_node.parent is None:
                    # We arrived at the root
                    break
                # Note that the root node is updated as well
                # We go up to the root in the tree
                current_node = current_node.parent

    def learn_one(self, x, y):
        # Setting current sample
        self._x = x
        self._y = y

        # Learning step
        leaf = self._go_downwards()
        if self.use_aggregation:
            self._go_upwards(leaf)

        # Incrementing iteration
        self.iteration += 1
        return self

    def predict_one(self, x):
        """Predict the label of the samples.

        Parameters
        ----------
        x
            Feature vector.

        """

        # If the tree hasn't seen any sample, then it should return
        # the default empty dict
        if not self._is_initialized:
            return

        leaf = (
            self._root.traverse(x, until_leaf=True)
            if isinstance(self._root, MondrianBranchRegressor)
            else self._root
        )

        if not self.use_aggregation:
            return self._predict(leaf)

        current = leaf
        prediction = 0.0

        while True:
            # This test is useless ?
            if isinstance(current, MondrianLeafRegressor):
                prediction = self._predict(current)
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = math.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                pred_new = self._predict(current)
                prediction = 0.5 * w * pred_new + (1 - 0.5 * w) * prediction

            # Root must be updated as well
            if current.parent is None:
                break

            # And now we go up
            current = current.parent

        return prediction
