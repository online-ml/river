from __future__ import annotations

from river import base
from river.tree.mondrian._mondrian_ops import (  # type: ignore[import-not-found]
    go_downwards_classifier_c,
    predict_proba_upward_c,
)
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import (
    MondrianBranchClassifier,
    MondrianLeafClassifier,
    MondrianNodeClassifier,
)


class MondrianTreeClassifier(MondrianTree, base.Classifier):
    """Mondrian Tree classifier.

    Parameters
    ----------
    step
        Step of the tree.
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    dirichlet
        Dirichlet parameter of the problem.
    split_pure
        Whether the tree should split pure leafs during training or not.
    iteration
        Number iterations to do during training.
    max_nodes
        Maximum number of nodes allowed in the tree. No new splits will occur once this
        limit is reached. If `None`, the tree grows without bound.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The Mondrian Tree Classifier is a type of decision tree that bases splitting decisions over a
    Mondrian process.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Bananas().take(500)

    >>> model = tree.mondrian.MondrianTreeClassifier(
    ...     step=0.1,
    ...     use_aggregation=True,
    ...     dirichlet=0.2,
    ...     seed=1
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 60.92%

    References
    ----------
    [^1]: Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh. Mondrian Forests: Efficient Online Random Forests.
        arXiv:1406.2673, pages 2-4

    """

    def __init__(
        self,
        step: float = 1.0,
        use_aggregation: bool = True,
        dirichlet: float = 0.5,
        split_pure: bool = False,
        iteration: int = 0,
        max_nodes: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            iteration=iteration,
            max_nodes=max_nodes,
            seed=seed,
        )

        self.dirichlet = dirichlet
        self.split_pure = split_pure

        # Training attributes
        # The previously observed classes set
        self._classes: set[base.typing.ClfTarget] = set()

        # Class-to-index mapping for list-based storage
        self._class_to_idx: dict = {}
        self._idx_to_class: list = []

        # Initialization of the root of the tree
        # It's the root so it doesn't have any parent (hence None)
        self._root = MondrianLeafClassifier(None, 0.0, 0)

    @property
    def _is_initialized(self):
        """Check if the tree has learnt at least one sample"""
        return len(self._classes) != 0

    def _predict(self, node: MondrianNodeClassifier) -> dict[base.typing.ClfTarget, float]:
        """Compute the prediction scores as a dict.

        Parameters
        ----------
        node
            Node to make predictions.

        """

        n_classes = len(self._classes)
        scores = node.predict(self.dirichlet, n_classes)
        return {self._idx_to_class[i]: scores[i] for i in range(n_classes)}

    def _split(
        self,
        node: MondrianLeafClassifier | MondrianBranchClassifier,
        split_time: float,
        threshold: float,
        feature: base.typing.FeatureName,
        is_right_extension: bool,
    ) -> MondrianBranchClassifier:
        """Split the given node and set the split time, threshold, etc., to the node.

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
        left: MondrianLeafClassifier | MondrianBranchClassifier
        right: MondrianLeafClassifier | MondrianBranchClassifier

        # The node is already a branch: we create a new branch above it and move the existing
        # node one level down the tree
        if isinstance(node, MondrianBranchClassifier):
            old_left, old_right = node.children
            if is_right_extension:
                left = MondrianBranchClassifier(
                    node, split_time, new_depth, node.feature, node.threshold
                )
                right = MondrianLeafClassifier(node, split_time, new_depth)
                left.replant(node, True)

                old_left.parent = left
                old_right.parent = left

                left.children = (old_left, old_right)
            else:
                right = MondrianBranchClassifier(
                    node, split_time, new_depth, node.feature, node.threshold
                )
                left = MondrianLeafClassifier(node, split_time, new_depth)
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
        branch = MondrianBranchClassifier(node.parent, node.time, node.depth, feature, threshold)
        left = MondrianLeafClassifier(branch, split_time, new_depth)
        right = MondrianLeafClassifier(branch, split_time, new_depth)
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

    def _go_downwards(self, x, y):
        """Update the tree (downward procedure)."""

        rng = self._rng

        leaf, new_root, nodes_added = go_downwards_classifier_c(
            self._root,
            x,
            self._class_to_idx[y],
            len(self._classes),
            self.dirichlet,
            self.use_aggregation,
            self.step,
            self.split_pure,
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

    def _go_upwards(self, leaf: MondrianLeafClassifier):
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

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        # Handle new class
        if y not in self._class_to_idx:
            self._class_to_idx[y] = len(self._idx_to_class)
            self._idx_to_class.append(y)

        # Updating the previously seen classes with the new sample
        self._classes.add(y)

        # Learning step
        leaf = self._go_downwards(x, y)
        if self.use_aggregation:
            self._go_upwards(leaf)

        # Incrementing iteration
        self.iteration += 1

    def predict_proba_one(self, x):
        """Predict the probability of the samples.

        Parameters
        ----------
        x
            Feature vector.

        """

        # If the tree hasn't seen any sample, then it should return
        # the default empty dict

        if not self._is_initialized:
            return {}

        # Find leaf using direct traversal, handling missing features
        current = self._root
        while not current.is_leaf:
            if current.feature in x:
                if x[current.feature] <= current.threshold:
                    current = current.children[0]
                else:
                    current = current.children[1]
            else:
                # Missing feature: follow the most traversed path
                _, current = current.most_common_path()

        if not self.use_aggregation:
            return self._predict(current)

        # Use Cython for the entire upward aggregation walk
        n_classes = len(self._classes)
        scores = predict_proba_upward_c(current, n_classes, self.dirichlet)

        return {self._idx_to_class[i]: scores[i] for i in range(n_classes)}
