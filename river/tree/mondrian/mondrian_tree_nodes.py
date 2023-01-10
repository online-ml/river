import abc
import math
import typing

from river.tree.base import Branch, Leaf
from river.utils.math import log_sum_2_exp


class MondrianLeaf(Leaf, abc.ABC):
    """Abstract class for all types of nodes in a Mondrian Tree.

    Parameters
    ----------
    parent
        Parent Node.
    n_features
        Number of features of the input data.
    time
        Split time of the node for Mondrian process.
    """

    def __init__(self, parent, n_features, time):
        super().__init__()

        # Generic Node attributes
        self.parent = parent
        self.n_features = n_features
        self.time = time

        self.is_leaf = True
        self.depth = 0
        self._left = None
        self._right = None
        self.feature = 0
        self.weight = 0.0
        self.log_weight_tree = 0.0
        self.threshold = 0.0
        self.n_samples = 0
        self.memory_range_min = [0.0 for _ in range(n_features)]
        self.memory_range_max = [0.0 for _ in range(n_features)]

    def copy(self, node):
        """Copy the node into the current one.

        Parameters
        ----------
        node
            Origin node to copy data from.
        """

        self.parent = node.parent
        self.time = node.time
        self.is_leaf = node.is_leaf
        self.depth = node.depth
        self.left = node.left
        self.right = node.right
        self.feature = node.feature
        self.weight = node.weight
        self.log_weight_tree = node.log_weight_tree
        self.threshold = node.threshold

    @abc.abstractmethod
    def _init_node(self, node) -> "MondrianLeaf":
        """Get the node and initialize it with default values if not
        yet initialized.

        Parameters
        ----------
        node
            Child node.
        """

    @property
    def left(self):
        if self._left is None:
            self._left = self._init_node(self._left)
        return self._left

    @left.setter
    def left(self, node):
        self._left = node

    @property
    def right(self):
        if self._right is None:
            self._right = self._init_node(self._right)
        return self._right

    @right.setter
    def right(self, node):
        self._right = node

    def update_depth(self, depth):
        """Update the depth of the current node with the given depth.

        Parameters
        ----------
        depth
            Depth of the node.

        Returns
        -------

        """
        depth += 1
        self.depth = depth

        # if it's a leaf, no need to update the children too
        if self.is_leaf:
            return

        # Updating the depth of the children as well
        self.left.update_depth(depth)
        self.right.update_depth(depth)

    def update_weight_tree(self):
        """Update the weight of the node in the tree."""
        if self.is_leaf:
            self.log_weight_tree = self.weight
        else:
            self.log_weight_tree = log_sum_2_exp(
                self.weight, self.left.log_weight_tree + self.right.log_weight_tree
            )

    # TODO: maybe there is a better name for this one, such as traverse, sort, or
    # something like that
    def get_child(self, x) -> "MondrianLeaf":
        """Get child node classifying x properly.

        Parameters
        ----------
        x
            Sample to find the path for.

        """
        if x[self.feature] <= self.threshold:
            return self.left
        else:
            return self.right

    @property
    def __repr__(self):
        return f"Node : {self.parent}, {self.time}"


class MondrianLeafClassifier(MondrianLeaf):
    """Mondrian Tree Classifier leaf.

    Parameters
    ----------
    parent
        Parent node.
    n_features
        Number of features of the problem.
    time
        Split time of the node.
    n_classes
        Number of classes of the problem.
    """

    def __init__(
        self,
        parent,
        n_features,
        time,
        n_classes,
    ):
        super().__init__(parent, n_features, time)
        self.n_classes = n_classes
        self.counts = [0 for _ in range(n_classes)]

    def _init_node(self, node):
        """Initialize a child node of the current one with the default values.

        Parameters
        ----------
        node
            Child node.
        """

        # Initialize the node with default values, at the right depth (depth + 1 since it's a child node)
        # This is mostly to have material to work with during computations, rather than handling the None
        # situation separately each time we encounter it
        node = MondrianLeafClassifier(self, self.n_features, 0, self.n_classes)
        node.depth = self.depth + 1

        return node

    def score(self, sample_class, dirichlet) -> float:
        """Computes the score of the node.

        Parameters
        ----------
        sample_class
            Class for which we want the score.
        dirichlet
            Dirichlet parameter of the tree.

        Notes
        -----
        This uses Jeffreys prior with Dirichlet parameter for smoothing.
        """

        count = self.counts[sample_class]
        n_classes = self.n_classes
        # We use the Jeffreys prior with dirichlet parameter
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(self, dirichlet) -> typing.Dict[int, float]:
        """Predict the scores of all classes and output a `scores` dictionary
        with the new values.

        Parameters
        ----------
        dirichlet
            Dirichlet parameter of the tree.
        """

        scores = {}
        for c in range(self.n_classes):
            scores[c] = self.score(c, dirichlet)
        return scores

    def loss(self, sample_class, dirichlet) -> float:
        """Compute the loss of the node.

        Parameters
        ----------
        sample_class
            A given class of the problem.
        dirichlet
            Dirichlet parameter of the problem.
        """

        sc = self.score(sample_class, dirichlet)
        return -math.log(sc)

    def update_weight(self, sample_class, dirichlet, use_aggregation, step) -> float:
        """Update the weight of the node given a class and the method used.

        Parameters
        ----------
        sample_class
            Class of a given sample.
        dirichlet
            Dirichlet parameter of the tree.
        use_aggregation
            Whether to use aggregation or not during computation (given by the tree).
        step
            Step parameter of the tree.
        """

        loss_t = self.loss(sample_class, dirichlet)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, sample_class):
        """Update the amount of samples that belong to a class in the node
        (not to use twice if you add one sample).

        Parameters
        ----------
        sample_class
            Class of a given sample.
        """

        self.counts[sample_class] += 1

    def is_dirac(self, sample_class):
        """Check whether the node follows a dirac distribution regarding the given
        class, i.e., if the node is pure regarding the given class.

        Parameters
        ----------
        sample_class
            Class of a given sample.
        """

        return self.n_samples == self.counts[sample_class]

    def update_downwards(
        self,
        x_t,
        sample_class,
        dirichlet,
        use_aggregation,
        step,
        do_update_weight,
    ):
        """Update the node when running a downward procedure updating the tree.

        Parameters
        ----------
        x_t
            Sample to proceed (as a list).
        sample_class
            Class of the sample x_t.
        dirichlet
            Dirichlet parameter of the tree.
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
            for j in range(self.n_features):
                x_tj = x_t[j]
                self.memory_range_min[j] = x_tj
                self.memory_range_max[j] = x_tj
        # Otherwise, we update the range
        else:
            for j in range(self.n_features):
                x_tj = x_t[j]
                if x_tj < self.memory_range_min[j]:
                    self.memory_range_min[j] = x_tj
                if x_tj > self.memory_range_max[j]:
                    self.memory_range_max[j] = x_tj

        # One more sample in the node
        self.n_samples += 1

        if do_update_weight:
            self.update_weight(sample_class, dirichlet, use_aggregation, step)

        self.update_count(sample_class)

    def range(self, feature_index) -> typing.Tuple[float, float]:
        """Output the known range of the node regarding the j-th feature.

        Parameters
        ----------
        feature_index
            Feature index for which you want to know the range.

        """
        return (
            self.memory_range_min[feature_index],
            self.memory_range_max[feature_index],
        )

    def range_extension(self, x_t, extensions):
        """Compute the range extension of the node for the given sample.

        Parameters
        ----------
        x_t
            Sample to deal with.
        extensions
            List of range extension per feature to update.
        """

        extensions_sum = 0.0
        for j in range(self.n_features):
            x_tj = x_t[j]
            feature_min_j, feature_max_j = self.range(j)
            if x_tj < feature_min_j:
                diff = feature_min_j - x_tj
            elif x_tj > feature_max_j:
                diff = x_tj - feature_max_j
            else:
                diff = 0
            extensions[j] = diff
            extensions_sum += diff
        return extensions_sum


# TODO: a leaf should be "promoted" to a branch. Right now, the branch acts simply
# as a wrapper.
class MondrianTreeBranch(Branch, abc.ABC):
    """A generic branch implementation for a Mondrian Tree.

    Parent and children are MondrianLeaf objects.

    Parameters
    ----------
    parent
        Origin node of the branch.
    """

    def __init__(self, parent):
        super().__init__((parent.left, parent.right))
        self.parent = parent

    def next(self, x):
        child = self.parent.get_child(x)
        if child.is_leaf:
            return child
        else:
            return MondrianTreeBranch(child)

    def most_common_path(self):
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError


class MondrianLeafRegressor(MondrianLeaf):
    def __init__(self, parent, n_features, time: float):
        super().__init__(parent, n_features, time)
        self.mean = 0.0

    def _init_node(self, node):
        """
        Get the child node and initialize if none exists.
        Parameters
        ----------
        node: MondrianTreeLeafRegressor
        Returns
        -------
        """
        node = MondrianLeafRegressor(self, self.n_features, 0)
        node.is_leaf = True
        node.depth = self.depth + 1
        return node

    def predict(self):
        """Returns the prediction of the node.
        Parameters
        ----------
        Returns
        -------
        Notes
        -----
        This uses Jeffreys prior with dirichlet parameter for smoothing
        """
        return self.mean

    def loss(self, sample_class):
        r = self.predict() - sample_class
        return r * r / 2

    def update_weight(self, sample_class, use_aggregation, step):
        loss_t = self.loss(sample_class)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_downwards(self, x_t, y_t, use_aggregation, step, do_update_weight):
        if self.n_samples == 0:
            for j in range(self.n_features):
                x_tj = x_t[j]
                self.memory_range_min[j] = x_tj
                self.memory_range_max[j] = x_tj
        else:
            for j in range(self.n_features):
                x_tj = x_t[j]
                if x_tj < self.memory_range_min[j]:
                    self.memory_range_min[j] = x_tj
                if x_tj > self.memory_range_max[j]:
                    self.memory_range_max[j] = x_tj

        self.n_samples += 1

        if do_update_weight:
            self.update_weight(y_t, use_aggregation, step)

        # Update the mean of the labels in the node online
        self.mean = (self.n_samples * self.mean + y_t) / (self.n_samples + 1)

    def range(self, j):
        return (
            self.memory_range_min[j],
            self.memory_range_max[j],
        )

    def range_extension(self, x_t, extensions):
        extensions_sum = 0
        for j in range(self.n_features):
            x_tj = x_t[j]
            feature_min_j, feature_max_j = self.range(j)
            if x_tj < feature_min_j:
                diff = feature_min_j - x_tj
            elif x_tj > feature_max_j:
                diff = x_tj - feature_max_j
            else:
                diff = 0
            extensions[j] = diff
            extensions_sum += diff
        return extensions_sum


# TODO: not sure this class is needed
class MondrianTreeBranchClassifier(MondrianTreeBranch):
    """A generic Mondrian Tree Branch for Classifiers.
    The specificity resides in the nature of the nodes which are all MondrianLeafClassifier instances.

    Parameters
    ----------
    parent
        Origin node of the tree
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def most_common_path(self):
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError


class MondrianTreeBranchRegressor(MondrianTreeBranch):
    """
    A generic Mondrian Tree Branch for Regressors.
    The specificity resides in the nature of the nodes which are all MondrianLeafRegressor instances.
    Parameters
    ----------
    parent
        Origin node of the tree
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def most_common_path(self):
        raise NotImplementedError

    @property
    def repr_split(self):
        raise NotImplementedError
