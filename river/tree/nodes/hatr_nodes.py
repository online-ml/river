import math
import typing

from river import base
from river import stats as st
from river.utils.random import poisson

from .branch import (
    DTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .htr_nodes import LeafAdaptive, LeafMean, LeafModel
from .leaf import HTLeaf


class NormError(base.Base):
    def __init__(self):
        # Normalized the input error
        self._norm = st.Var()
        # Monitor the normalized mean error values
        self._mean = st.Mean()

    def push(self, drift_input: float) -> float:
        # Update the variance estimator to get the updated estimated range
        self._norm.update(drift_input)

        if self.length == 1:
            # The expected error is the normalized (MinMax) mean error
            norm_input = 0.5
        else:
            sd = math.sqrt(self._norm.get())

            # We assume the error follows a normal distribution -> (empirical rule) 99.73% of the values
            # lie  between [mean - 3*sd, mean + 3*sd]. We assume this range for the normalized data.
            # Hence, we can apply the min-max norm to cope with ADWIN's requirements
            norm_input = (drift_input + 3 * sd) / (6 * sd) if sd > 0 else 0.5

        self._mean.update(norm_input)

        return norm_input

    @property
    def mean_error(self) -> float:
        return self._mean.get()

    @property
    def length(self) -> int:
        return self._norm.mean.n


class AdaLeafRegressor(HTLeaf):
    """Learning Node of the Hoeffding Adaptive Tree regressor.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the learning node in the tree.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    drift_detector
        The detector used to monitor concept drifts.
    rng
        Random number generator used in Poisson sampling.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)

        self.drift_detector = drift_detector
        self.rng = rng

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._normalizer = NormError()

    def kill_tree_children(self, hatr):
        pass

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None):
        y_pred = self.prediction(x, tree=tree)

        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = poisson(rate=1, rng=self.rng)
            if k > 0:
                sample_weight *= k

        old_error = self._normalizer.mean_error
        normalized_error = self._normalizer.push(abs(y - y_pred))
        # Update the drift detector
        self.drift_detector.update(normalized_error)
        error_change = self.drift_detector.drift_detected

        # Error is decreasing
        if error_change and self._normalizer.mean_error < old_error:
            # Reset the error estimator
            self._normalizer = self._normalizer.clone()
            error_change = False

        # Update learning model
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

        weight_seen = self.total_weight

        if weight_seen - self.last_split_attempt_at >= tree.grace_period:
            if self.depth >= tree.max_depth:
                # Depth-based pre-pruning
                self.deactivate()
                tree._n_inactive_leaves += 1
                tree._n_active_leaves -= 1
            elif self.is_active():
                tree._attempt_to_split(
                    self,
                    parent,
                    parent_branch,
                    drift_detector=tree.drift_detector.clone(),
                )
                self.last_split_attempt_at = weight_seen


class AdaBranchRegressor(DTBranch):
    """Node that splits the data in a Hoeffding Adaptive Tree Regression.

    Parameters
    ----------
    stats
        Target stats.
    depth
        The depth of the node.
    drift_detector
        The detector used to monitor concept drifts.
    attributes
        Other parameters passed to the split node.
    """

    def __init__(self, stats, *children, drift_detector, **attributes):
        stats = stats if stats else st.Var()
        super().__init__(stats, *children, **attributes)
        self.drift_detector = drift_detector
        self._alternate_tree = None

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._normalizer = NormError()

    def traverse(self, x, until_leaf=True) -> typing.List[HTLeaf]:  # type: ignore
        """Return the leaves corresponding to the given input.

        Alternate subtree leaves are also included.

        Parameters
        ----------
        x
            The input instance.
        until_leaf
            Whether or not branch nodes can be returned in case of missing features or emerging
            feature categories.
        """
        found_nodes: typing.List[HTLeaf] = []
        for node in self.walk(x, until_leaf=until_leaf):
            if isinstance(node, AdaBranchRegressor) and node._alternate_tree:
                if isinstance(node._alternate_tree, AdaBranchRegressor):
                    found_nodes.append(
                        node._alternate_tree.traverse(x, until_leaf=until_leaf)  # type: ignore
                    )
                else:
                    found_nodes.append(node._alternate_tree)  # type: ignore

        found_nodes.append(node)  # type: ignore
        return found_nodes

    def iter_leaves(self):
        """Iterate over leaves from the left-most one to the right-most one.

        Overrides the base implementation by also including alternate subtrees.
        """
        for child in self.children:
            yield from child.iter_leaves()

            if isinstance(child, AdaBranchRegressor) and child._alternate_tree:
                yield from child._alternate_tree.iter_leaves()

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None):
        leaf = super().traverse(x, until_leaf=True)
        y_pred = leaf.prediction(x, tree=tree)

        # Update stats as traverse the tree to improve predictions (in case split nodes are used
        # to provide responses)
        self.stats.update(y, sample_weight)

        old_error = self._normalizer.mean_error
        normalized_error = self._normalizer.push(abs(y - y_pred))

        self.drift_detector.update(normalized_error)
        error_change = self.drift_detector.drift_detected

        # Error is decreasing, better to keep things as they are
        if error_change and self._normalizer.mean_error < old_error:
            # Reset the error estimator
            self._normalizer = self._normalizer.clone()
            error_change = False

        # Condition to build a new alternate tree
        if error_change:
            # Reset the error estimator
            self._normalizer = self._normalizer.clone()
            self._alternate_tree = tree._new_leaf(parent=self)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1

        # Condition to replace alternate tree
        elif self._alternate_tree:
            alt_length = self._alternate_tree._normalizer.length
            curr_length = self._normalizer.length
            if (
                alt_length > tree.drift_window_threshold
                and curr_length > tree.drift_window_threshold
            ):
                old_error_rate = self._normalizer.mean_error
                alt_error_rate = self._alternate_tree._normalizer.mean_error

                f_delta = 0.05
                f_n = 1.0 / alt_length + 1.0 / curr_length

                try:
                    bound = math.sqrt(
                        2.0
                        * old_error_rate
                        * (1.0 - old_error_rate)
                        * math.log(2.0 / f_delta)
                        * f_n
                    )
                except ValueError:  # error rate exceeds 1, so we clip it
                    bound = 0.0
                if bound < (old_error_rate - alt_error_rate):
                    tree._n_active_leaves -= self.n_leaves
                    tree._n_active_leaves += self._alternate_tree.n_leaves
                    self.kill_tree_children(tree)

                    if parent is not None:
                        parent.children[parent_branch] = self._alternate_tree
                        self._alternate_tree = None
                    else:
                        # Switch tree root
                        tree._root = tree._root._alternate_tree
                    tree._n_switch_alternate_trees += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, DTBranch):
                        self._alternate_tree.kill_tree_children(tree)  # noqa
                    self._alternate_tree = None
                    tree._n_pruned_alternate_trees += 1

        # Learn one sample in alternate tree and child nodes
        if self._alternate_tree:
            self._alternate_tree.learn_one(
                x,
                y,
                sample_weight=sample_weight,
                tree=tree,
                parent=parent,
                parent_branch=parent_branch,
            )
        try:
            child = self.next(x)
        except KeyError:
            child = None

        if child is not None:
            child.learn_one(
                x,
                y,
                sample_weight=sample_weight,
                tree=tree,
                parent=self,
                parent_branch=self.branch_no(x),
            )
        else:
            # Instance contains a categorical value previously unseen by the split node
            if self.max_branches() == -1 and self.feature in x:  # noqa
                # Creates a new learning node to encompass the new observed feature value
                leaf = tree._new_leaf(parent=self)
                self.add_child(x[self.feature], leaf)  # noqa
                tree._n_active_leaves += 1
                leaf.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=tree,
                    parent=self,
                    parent_branch=self.branch_no(x),
                )
            # The split feature is missing in the instance. Hence, we pass the new example
            # to the most traversed path in the current subtree
            else:
                child_id, child = self.most_common_path()
                child.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=tree,
                    parent=self,
                    parent_branch=child_id,
                )

    # Override AdaNode
    def kill_tree_children(self, tree):
        for child in self.children:
            # Delete alternate tree if it exists
            if isinstance(child, DTBranch):
                if child._alternate_tree:
                    child._alternate_tree.kill_tree_children(tree)
                    tree._n_pruned_alternate_trees += 1
                    child._alternate_tree = None

                # Recursive delete of SplitNodes
                child.kill_tree_children(tree)  # noqa
            else:
                if child.is_active():  # noqa
                    tree._n_active_leaves -= 1
                else:
                    tree._n_inactive_leaves -= 1


class AdaNomBinaryBranchReg(AdaBranchRegressor, NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)


class AdaNumBinaryBranchReg(AdaBranchRegressor, NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)


class AdaNomMultiwayBranchReg(AdaBranchRegressor, NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)


class AdaNumMultiwayBranchReg(AdaBranchRegressor, NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **attributes):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **attributes)


class AdaLeafRegMean(AdaLeafRegressor, LeafMean):
    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, drift_detector, rng, **kwargs)


class AdaLeafRegModel(AdaLeafRegressor, LeafModel):
    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, drift_detector, rng, **kwargs)


class AdaLeafRegAdaptive(AdaLeafRegressor, LeafAdaptive):
    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, drift_detector, rng, **kwargs)
