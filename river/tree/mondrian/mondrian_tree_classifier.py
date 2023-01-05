import math

from random import uniform
from random import choices

from river.tree.mondrian.mondrian_tree import MondrianTree

from river.tree.mondrian.mondrian_tree_nodes import MondrianTreeBranchClassifier
from river.tree.mondrian.mondrian_tree_nodes import MondrianTreeLeafClassifier

from river.base import typing


class MondrianTreeClassifier(MondrianTree):
    """
    Mondrian Tree classifier.

    https://proceedings.neurips.cc/paper/2014/file/d1dc3a8270a6f9394f88847d7f0050cf-Paper.pdf

    Parameters
    ----------
    n_classes:
        Number of classes of the problem
    n_features:
        Number of features of the data in entry
    step:
        Step of the tree
    use_aggregation:
        Whether to use aggregation weighting techniques or not.
    dirichlet:
        Dirichlet parameter of the problem
    split_pure:
        Whether the tree should split pure leafs during training or not
    iteration:
        Number iterations to do during training
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
    ):

        super().__init__(
            n_features=n_features,
            step=step,
            loss="log",
            use_aggregation=use_aggregation,
            split_pure=split_pure,
            iteration=iteration,
        )
        self.n_classes = n_classes
        self.dirichlet = dirichlet

        # Initialization of the root of the tree
        # It's the root so it doesn't have any parent (hence None)
        self.tree = MondrianTreeBranchClassifier(MondrianTreeLeafClassifier(None, self.n_features, 0.0, self.n_classes))

        # Training attributes
        # The previously observed classes dictionary: enables to convert the class label into a positive integer
        self._classes = {}

        # The current sample being proceeded
        self._x = None
        # The current label index being proceeded
        self._y = None

    def score(self, node: MondrianTreeLeafClassifier) -> float:
        """
        Computes the score of the node regarding the current sample being proceeded

        Parameters
        ----------
        node
            Node to evaluate the score of

        """
        return node.score(self._y, self.dirichlet)

    def predict(self, node: MondrianTreeLeafClassifier) -> dict[float]:
        """
        Computes the predictions scores of the node regarding all the classes scores.

        Parameters
        ----------
        node
            Node to make predictions with

        """
        return node.predict(self.dirichlet)

    def loss(self, node: MondrianTreeLeafClassifier) -> float:
        """
        Computes the loss for the given node regarding the current label

        Parameters
        ----------
        node
            Node to evaluate the loss for
        """
        return node.loss(self._y, self.dirichlet)

    def update_weight(self, node: MondrianTreeLeafClassifier) -> float:
        """
        Updates the weight of the node regarding the current label with the tree parameters

        Parameters
        ----------
        node
            Node to update the weight of

        """
        return node.update_weight(self._y, self.dirichlet, self.use_aggregation, self.step)

    def update_count(self, node: MondrianTreeLeafClassifier):
        """
        Update the count of labels with the current class `_y` being treated (not to use twice for one sample added)
        Parameters
        ----------
        node
            Target node

        """
        return node.update_count(self._y)

    def update_downwards(self, node: MondrianTreeLeafClassifier, do_update_weight: bool):
        """
        Updates the node when running a downward procedure updating the tree

        Parameters
        ----------
        node
            Target node
        do_update_weight
            Whether we should update the weights or not

        """
        return node.update_downwards(self._x, self._y, self.dirichlet, self.use_aggregation, self.step,
                                     do_update_weight)

    def compute_split_time(self, node: MondrianTreeLeafClassifier) -> float:
        """
        Computes the spit time of the given node

        Parameters
        ----------
        node: MondrianTreeLeafClassifier

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
                T = float('inf')
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

    def split(
            self,
            node: MondrianTreeLeafClassifier,
            split_time: float,
            threshold: float,
            feature: int,
            is_right_extension: bool
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
            left = child_node = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, left, right)

        else:
            right = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            left = MondrianTreeLeafClassifier(node, self.n_features, split_time, self.n_classes)
            extend_child(node, right, left)

        # Updating current node parameters
        node.left = left
        node.right = right
        node.feature = feature
        node.threshold = threshold
        node.is_leaf = False

    def go_downwards(self):
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
            self.update_downwards(current_node, False)
            return current_node
        else:
            while True:
                # If it's not the first iteration (otherwise the current node
                # is root with no range), we consider the possibility of a split
                split_time = self.compute_split_time(current_node)

                if split_time > 0:
                    # We split the current node: because the current node is a
                    # leaf, or because we add a new node along the path

                    # We normalize the range extensions to get probabilities
                    intensities_sum = math.fsum(self.intensities)
                    for i in range(len(self.intensities)):
                        self.intensities[i] /= intensities_sum

                    # Sample the feature at random with a probability
                    # proportional to the range extensions
                    feature = choices(list(range(self.n_features)), self.intensities, k=1)[0]
                    x_tf = self._x[feature]

                    # Is it a right extension of the node ?
                    range_min, range_max = current_node.range(feature)
                    is_right_extension = x_tf > range_max
                    if is_right_extension:
                        threshold = uniform(range_max, x_tf)
                    else:
                        threshold = uniform(x_tf, range_min)

                    # We split the current node
                    self.split(
                        current_node,
                        split_time,
                        threshold,
                        feature,
                        is_right_extension,
                    )

                    # Update the current node
                    self.update_downwards(current_node, True)

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
                    self.update_downwards(leaf, False)
                    return leaf
                else:
                    # There is no split, so we just update the node and go to the next one
                    self.update_downwards(current_node, True)
                    if current_node.is_leaf:
                        return current_node
                    else:
                        current_node = current_node.get_child(self._x)

    def go_upwards(self, leaf: MondrianTreeLeafClassifier):
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

    def learn_one(self, x: dict, y: typing.ClfTarget):
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
        leaf = self.go_downwards()
        if self.use_aggregation:
            self.go_upwards(leaf)

        # Incrementing iteration
        self.iteration += 1
        return self

    @property
    def _multiclass(self):
        return True

    def find_leaf(self, x: dict) -> MondrianTreeLeafClassifier:
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
        while not is_leaf:
            is_leaf = node.is_leaf
            if not is_leaf:
                feature_index = node.feature
                feature = list(x.keys())[feature_index]
                threshold = node.threshold
                if x[feature] <= threshold:
                    node = node.left
                else:
                    node = node.right
        return node

    def predict_proba_one(self, x: dict):
        """
        Predict the probability of the samples
        Parameters
        ----------
        x
            Feature vector
        """

        # We turn the indexes into the labels names
        classes_name = list(self._classes.keys())

        def set_scores(values: list):
            """Turns the list of score values into the dictionary of scores output"""
            for k in range(self.n_classes):
                scores[classes_name[k]] = values[k]

        # Initialization of the scores to output to 0
        scores = {}
        set_scores([0] * self.n_features)

        leaf = self.find_leaf(x)

        if not self.use_aggregation:
            predictions = self.predict(leaf)
            set_scores(list(predictions.values()))
            return scores

        current = leaf

        while True:
            # This test is useless ?
            if current.is_leaf:
                predictions = self.predict(current)
                set_scores(list(predictions.values()))
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = math.exp(weight - log_weight_tree)
                # Get the predictions of the current node
                pred_new = self.predict(current)
                for i in range(self.n_classes):
                    label = classes_name[i]
                    scores[label] = 0.5 * w * pred_new[i] + (1 - 0.5 * w) * scores[label]

            # Root must be updated as well
            if current.parent is None:
                break
            # And now we go up
            current = current.parent

        return scores
