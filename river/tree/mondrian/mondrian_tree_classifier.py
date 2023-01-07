import math
import sys

from river import base
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import (
    MondrianLeafClassifier,
    MondrianTreeBranchClassifier,
)


class MondrianTreeClassifier(MondrianTree):
    """
    Mondrian Tree classifier.

    Parameters
    ----------
    n_classes
        Number of classes of the problem
    n_features
        Number of features of the data in entry
    step
        Step of the tree
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    dirichlet
        Dirichlet parameter of the problem
    split_pure
        Whether the tree should split pure leafs during training or not
    iteration
        Number iterations to do during training
    seed
        Random seed for reproducibility

    Notes
    -----
    The Mondrian Tree Classifier is a type of decision tree that bases splitting decisions over a Mondrian process.

    References
    ----------
    [^1] Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh. Mondrian Forests: Efficient Online Random Forests.
        arXiv:1406.2673, pages 2-4
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_features: int = 1,
        step: float = 0.1,
        use_aggregation: bool = True,
        dirichlet: float = None,
        split_pure: bool = False,
        iteration: int = 0,
        seed: int = None,
    ):

        super().__init__(
            n_features=n_features,
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            iteration=iteration,
            seed=seed,
        )
        self.n_classes = n_classes
        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        # Training attributes
        # The previously observed classes dictionary: enables to convert the class label into a positive integer
        self._classes: dict[base.typing.ClfTarget, int] = {}

        # The current sample being proceeded
        self._x: list[float] = []
        # The current label index being proceeded
        self._y: int = -1

        # Initialization of the root of the tree
        # It's the root so it doesn't have any parent (hence None)
        self.tree = MondrianTreeBranchClassifier(
            MondrianLeafClassifier(None, n_features, 0.0, n_classes)
        )

    def _score(self, node) -> float:
        """
        Computes the score of the node regarding the current sample being proceeded

        Parameters
        ----------
        node
            Node to evaluate the score of

        """
        return node.score(self._y, self.dirichlet)

    def _predict(self, node) -> dict[int, float]:
        """
        Computes the predictions scores of the node regarding all the classes scores.

        Parameters
        ----------
        node
            Node to make predictions with

        """
        return node.predict(self.dirichlet)

    def _loss(self, node) -> float:
        """
        Computes the loss for the given node regarding the current label

        Parameters
        ----------
        node
            Node to evaluate the loss for
        """
        return node.loss(self._y, self.dirichlet)

    def _update_weight(self, node) -> float:
        """
        Updates the weight of the node regarding the current label with the tree parameters

        Parameters
        ----------
        node
            Node to update the weight of

        """
        return node.update_weight(self._y, self.dirichlet, self.use_aggregation, self.step)

    def _update_count(self, node):
        """
        Update the count of labels with the current class `_y` being treated (not to use twice for one sample added)
        Parameters
        ----------
        node
            Target node

        """
        return node.update_count(self._y)

    def _update_downwards(self, node, do_update_weight):
        """
        Updates the node when running a downward procedure updating the tree

        Parameters
        ----------
        node
            Target node
        do_update_weight
            Whether we should update the weights or not

        """
        return node.update_downwards(
            self._x, self._y, self.dirichlet, self.use_aggregation, self.step, do_update_weight
        )

    def _compute_split_time(self, node):
        """
        Computes the spit time of the given node

        Parameters
        ----------
        node
            Target node

        """
        #  Don't split if the node is pure: all labels are equal to the one of y_t
        if not self.split_pure and node.is_dirac(self._y):
            return 0.0

        extensions_sum = node.range_extension(self._x, self.intensities)
        # If x_t extends the current range of the node
        if extensions_sum > 0:
            # Sample an exponential with intensity = extensions_sum
            # try catch to handle the Overflow situation in the exponential
            try:
                T = math.exp(1 / extensions_sum)
            except OverflowError:
                T = sys.float_info.max  # we get the largest possible output instead
            time = node.time
            # Splitting time of the node (if splitting occurs)
            split_time = time + T
            # If the node is a leaf we must split it
            if node.is_leaf:
                return split_time
            # Otherwise we apply Mondrian process dark magic :)
            # 1. We get the creation time of the childs (left and right is the same)
            left = node.left
            child_time = left.time
            # 2. We check if splitting time occurs before child creation time
            if split_time < child_time:
                return split_time

        return 0

    def _split(
        self,
        node: MondrianLeafClassifier,
        split_time: float,
        threshold: float,
        feature: int,
        is_right_extension: bool,
    ):
        """
        Splits the given node and attributes the split time, threshold, etc... to the node

        Parameters
        ----------
        node
            Target node
        split_time
            Split time of the node in the Mondrian process
        threshold
            Threshold of acceptance of the node
        feature
            Feature index of the node
        is_right_extension
            Should we extend the tree in the right or left direction
        """

        # Let's build a function to handle splitting depending on what side we go into
        def extend_child(parent, main_child, other):
            # Expending the node towards the main child chosen
            main_child.copy(parent)
            main_child.parent = parent
            main_child.time = split_time

            # other must have node has parent
            other.is_leaf = True

            # If the node previously had children, we have to update it
            if not parent.is_leaf:
                old_left = parent.left
                old_right = parent.right
                old_right.parent = main_child
                old_left.parent = main_child

        # Create the two splits
        if is_right_extension:
            left = MondrianLeafClassifier(node, self.n_features, split_time, self.n_classes)
            right = MondrianLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, left, right)

        else:
            right = MondrianLeafClassifier(node, self.n_features, split_time, self.n_classes)
            left = MondrianLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, right, left)

        # Updating current node parameters
        node.left = left
        node.right = right
        node.feature = feature
        node.threshold = threshold
        node.is_leaf = False

    def _go_downwards(self):
        """
        Updates the tree (downward procedure)

        """

        # We update the nodes along the path which leads to the leaf containing the current sample.
        # For each node on the path, we consider the possibility of
        # splitting it, following the Mondrian process definition.

        # We start at the root
        current_node = self.tree.parent

        if self.iteration == 0:
            # If it's the first iteration, we just put the current sample in the range of root
            self._update_downwards(current_node, False)
            return current_node
        else:
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = self._compute_split_time(current_node)

                if split_time > 0:
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path

                    # We normalize the range extensions to get probabilities
                    intensities_sum = math.fsum(self.intensities)
                    for i in range(len(self.intensities)):
                        self.intensities[i] /= intensities_sum

                    # Sample the feature at random with a probability
                    # proportional to the range extensions
                    feature = self.random_generator.choices(
                        list(range(self.n_features)), self.intensities, k=1
                    )[0]
                    x_tf = self._x[feature]

                    # Is it a right extension of the node ?
                    range_min, range_max = current_node.range(feature)
                    is_right_extension = x_tf > range_max
                    if is_right_extension:
                        threshold = self.random_generator.uniform(range_max, x_tf)
                    else:
                        threshold = self.random_generator.uniform(x_tf, range_min)

                    # We split the current node
                    self._split(
                        current_node, split_time, threshold, feature, is_right_extension,
                    )

                    # Update the current node
                    self._update_downwards(current_node, True)

                    left = current_node.left
                    right = current_node.right
                    depth = current_node.depth

                    # Now, get the next node
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left

                    # We update the depth of each child
                    left.update_depth(depth)
                    right.update_depth(depth)

                    # This is the leaf containing the sample point (we've just
                    # splitted the current node with the data point)
                    leaf = current_node
                    self._update_downwards(leaf, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to the next one
                    self._update_downwards(current_node, True)
                    if current_node.is_leaf:
                        return current_node
                    else:
                        current_node = current_node.get_child(self._x)

    def _go_upwards(self, leaf):
        """
        Updates the tree (upwards procedure)

        Parameters
        ----------
        leaf
            Leaf to start from when going upward

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

    def _find_leaf(self, x):
        """
        Finds the leaf that contains the sample, starting from the root

        Parameters
        ----------
        x
            Feature vector
        """

        # We start at the root
        node = self.tree.parent

        is_leaf = False
        features = list(x.keys())
        while not is_leaf:
            is_leaf = node.is_leaf
            if not is_leaf:
                feature_index = node.feature
                feature = features[feature_index]
                threshold = node.threshold
                if x[feature] <= threshold:
                    node = node.left
                else:
                    node = node.right
        return node

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        """
        Learns the sample (x, y)

        Parameters
        ----------
        x
            Feature vector of the sample
        y
            Label of the sample
        """

        # Updating the previously seen classes with the new sample
        if y not in self._classes:
            self._classes[y] = len(self._classes)

        # Setting current sample
        # We change x in a list here to make computations easier afterwards
        self._x = list(x.values())
        # Current sample label index
        self._y = self._classes[y]

        # Learning step
        leaf = self._go_downwards()
        if self.use_aggregation:
            self._go_upwards(leaf)

        # Incrementing iteration
        self.iteration += 1
        return self

    def predict_proba_one(self, x):
        """
        Predict the probability of the samples

        Parameters
        ----------
        x
            Feature vector
        """

        # We turn the indexes into the labels names
        classes_name = list(self._classes.keys())

        def set_scores(values: list[float]):
            """Turns the list of score values into the dictionary of scores output"""
            for k in range(self.n_classes):
                scores[classes_name[k]] = values[k]

        # Initialization of the scores to output to 0
        scores: dict[base.typing.ClfTarget, float] = {}
        set_scores([0.0 for _ in range(self.n_features)])

        leaf = self._find_leaf(x)

        if not self.use_aggregation:
            predictions = self._predict(leaf)
            set_scores(list(predictions.values()))
            return scores

        current = leaf

        while True:
            # This test is useless ?
            if current.is_leaf:
                predictions = self._predict(current)
                set_scores(list(predictions.values()))
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = math.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                pred_new = self._predict(current)
                for i in range(self.n_classes):
                    label = classes_name[i]
                    scores[label] = 0.5 * w * pred_new[i] + (1 - 0.5 * w) * scores[label]

            # Root must be updated as well
            if current.parent is None:
                break

            # And now we go up
            current = current.parent

        return scores
