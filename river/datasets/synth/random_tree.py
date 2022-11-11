from __future__ import annotations

import numpy as np

from river import datasets
from river.utils.skmultiflow_utils import check_random_state


class RandomTree(datasets.base.SyntheticDataset):
    """Random Tree generator.

    This generator is based on [^1]. The generator creates a random
    tree by splitting features at random and setting labels at its leaves.

    The tree structure is composed of node objects, which can be either inner
    nodes or leaf nodes. The choice comes as a function of the parameters
    passed to its initializer.

    Since the concepts are generated and classified according to a tree
    structure, in theory, it should favor decision tree learners.

    Parameters
    ----------
    seed_tree
        Seed for random generation of tree.
    seed_sample
        Seed for random generation of instances.
    n_classes
        The number of classes to generate.
    n_num_features
        The number of numerical features to generate.
    n_cat_features
        The number of categorical features to generate.
    n_categories_per_feature
        The number of values to generate per categorical feature.
    max_tree_depth
        The maximum depth of the tree concept.
    first_leaf_level
        The first level of the tree above `max_tree_depth` that can have leaves.
    fraction_leaves_per_level
        The fraction of leaves per level from `first_leaf_level` onwards.

    Examples
    --------

    >>> from river.datasets import synth

    >>> dataset = synth.RandomTree(seed_tree=42, seed_sample=42, n_classes=2,
    ...                            n_num_features=2, n_cat_features=2,
    ...                            n_categories_per_feature=2, max_tree_depth=6,
    ...                            first_leaf_level=3, fraction_leaves_per_level=0.15)

    >>> for x, y in dataset.take(5):
    ...     print(x, y)
    {'x_num_0': 0.3745, 'x_num_1': 0.9507, 'x_cat_0': 0, 'x_cat_1': 1} 1
    {'x_num_0': 0.5986, 'x_num_1': 0.1560, 'x_cat_0': 0, 'x_cat_1': 0} 1
    {'x_num_0': 0.0580, 'x_num_1': 0.8661, 'x_cat_0': 1, 'x_cat_1': 1} 0
    {'x_num_0': 0.7080, 'x_num_1': 0.0205, 'x_cat_0': 1, 'x_cat_1': 1} 0
    {'x_num_0': 0.8324, 'x_num_1': 0.2123, 'x_cat_0': 1, 'x_cat_1': 1} 0

    References
    ----------
    [^1]: Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams."
          In Proceedings of the sixth ACM SIGKDD international conference on
          Knowledge discovery and data mining, pp. 71-80. 2000.

    """

    def __init__(
        self,
        seed_tree: int | np.random.RandomState | None = None,
        seed_sample: int | np.random.RandomState | None = None,
        n_classes: int = 2,
        n_num_features: int = 5,
        n_cat_features: int = 5,
        n_categories_per_feature: int = 5,
        max_tree_depth: int = 5,
        first_leaf_level: int = 3,
        fraction_leaves_per_level: float = 0.15,
    ):
        super().__init__(
            n_features=n_num_features + n_cat_features,
            n_classes=n_classes,
            n_outputs=1,
            task=datasets.base.MULTI_CLF,
        )

        self.seed_tree = seed_tree
        self.seed_sample = seed_sample
        self.n_num_features = n_num_features
        self.n_cat_features = n_cat_features
        self.n_categories_per_feature = n_categories_per_feature
        self.max_tree_depth = max_tree_depth
        self.first_leaf_level = first_leaf_level
        self.fraction_leaves_per_level = fraction_leaves_per_level
        self.tree_root = None

        self.features_num = [f"x_num_{i}" for i in range(self.n_num_features)]
        self.features_cat = [f"x_cat_{i}" for i in range(self.n_cat_features)]
        self.feature_names = self.features_num + self.features_cat
        self.target_values = [i for i in range(self.n_classes)]

    def _generate_random_tree(self):
        """
        Generates the random tree, starting from the root node and following
        the constraints passed as parameters to the initializer.

        The tree is recursively generated, node by node, until it reaches the
        maximum tree depth.
        """
        rng_tree = check_random_state(self.seed_tree)
        candidate_features = np.arange(self.n_num_features + self.n_cat_features)
        min_numeric_values = np.zeros(self.n_num_features)
        max_numeric_values = np.ones(self.n_num_features)

        self.tree_root = self._generate_random_tree_node(
            0, candidate_features, min_numeric_values, max_numeric_values, rng_tree
        )

    def _generate_random_tree_node(
        self,
        current_depth: int,
        candidate_features: np.ndarray,
        min_numeric_value: np.ndarray,
        max_numeric_value: np.ndarray,
        rng: np.random.RandomState,
    ):
        """
        Creates a node, choosing at random the splitting feature and value.
        Then recursively generates its children. If the split feature is a
        numerical feature there will be two children nodes (left and right),
        one for samples where the value for the split feature is smaller than
        the split value, and one for the other case. For categorical features,
        the number of children generated is equal to the number of categories
        per categorical feature.

        Once the recursion passes the leaf_node minimum depth, it probabilistic
        chooses if the node is a leaf_node or not. If not, the recursion follow
        the same way as before. If it decides the node is a leaf_node, a class
        label is chosen for the leaf_node at random.

        Finally, if the current_depth is equal or higher than the tree
        maximum depth, a leaf_node node is immediately returned.

        Parameters
        ----------
        current_depth
            The current tree depth.
        candidate_features
            Candidate features
        min_numeric_value
            The minimum numeric feature value, on this branch of the tree.
        max_numeric_value
            The minimum numeric feature value, on this branch of the tree.
        rng
            A numpy random number generator instance.

        Notes
        -----
        If the splitting attribute of a node happens to be a nominal attribute
        we guarantee that none of its children will split on the same attribute,
        as it would have no use for that split.

        """
        # Stop recursive call
        if (current_depth >= self.max_tree_depth) or (
            (current_depth >= self.first_leaf_level)
            and (self.fraction_leaves_per_level >= (1.0 - rng.rand()))
        ):
            leaf_node = TreeNode()
            leaf_node.class_label = rng.randint(self.n_classes)
            return leaf_node
        # Create a new node
        split_node = TreeNode()
        chosen_feature = rng.randint(candidate_features.size)
        if chosen_feature < self.n_num_features:
            # Chosen feature is numeric
            split_node.split_feature_idx = chosen_feature
            min_val = min_numeric_value[chosen_feature]
            max_val = max_numeric_value[chosen_feature]
            split_node.split_feature_val = (max_val - min_val) * rng.rand() + min_val
            # Left node
            new_max_value = np.array(max_numeric_value)
            new_max_value[chosen_feature] = split_node.split_feature_val
            split_node.children.append(
                self._generate_random_tree_node(
                    current_depth + 1,
                    candidate_features,
                    min_numeric_value,
                    new_max_value,
                    rng,
                )
            )
            # Right node
            new_min_value = np.array(min_numeric_value)
            new_min_value[chosen_feature] = split_node.split_feature_val
            split_node.children.append(
                self._generate_random_tree_node(
                    current_depth + 1,
                    candidate_features,
                    new_min_value,
                    max_numeric_value,
                    rng,
                )
            )
        else:
            # Chosen feature is categorical
            split_node.split_feature_idx = candidate_features[chosen_feature]
            # Remove chosen features from candidates
            new_candidates = np.delete(
                candidate_features,
                np.argwhere(candidate_features == split_node.split_feature_idx),
            )
            # Generate children per category
            for i in range(self.n_categories_per_feature):
                split_node.children.append(
                    self._generate_random_tree_node(
                        current_depth + 1,
                        new_candidates,
                        min_numeric_value,
                        max_numeric_value,
                        rng,
                    )
                )
        return split_node

    def _classify_instance(self, node, x):
        if len(node.children) == 0:
            # Reached a leaf node
            return node.class_label
        feature = self.feature_names[node.split_feature_idx]
        if node.split_feature_idx < self.n_num_features:
            idx = 0 if x[feature] < node.split_feature_val else 1
            return self._classify_instance(node.children[idx], x)
        else:
            idx = x[feature]
            return self._classify_instance(node.children[idx], x)

    def __iter__(self):
        rng_sample = check_random_state(self.seed_sample)
        # Generate random tree model which will be used to classify instances
        self._generate_random_tree()

        # Randomly generate features, and then classify the resulting instance.
        while True:
            x = dict()
            for feature in self.features_num:
                x[feature] = rng_sample.rand()
            for feature in self.features_cat:
                x[feature] = rng_sample.randint(self.n_categories_per_feature)
            y = self._classify_instance(self.tree_root, x)
            yield x, y


class TreeNode:
    """Class that stores the attributes of a tree node.

    Parameters
    ----------
    class_label
        Class label if the node is a leaf node.

    split_feature_idx
        Feature index for the split, if the node is a split node.

    split_feature_val
        Feature value for the split, if the node is a split node.
    """

    def __init__(
        self,
        class_label: int = None,
        split_feature_idx: int = None,
        split_feature_val: int | float | None = None,
    ):
        self.class_label = class_label
        self.split_feature_idx = split_feature_idx
        self.split_feature_val = split_feature_val
        self.children: list = []
