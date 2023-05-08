from __future__ import annotations

import math

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

        self._error_tracker = st.Var()

    def kill_tree_children(self, hatr):
        pass

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None):
        y_pred = self.prediction(x, tree=tree)

        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = poisson(rate=1, rng=self.rng)
            if k > 0:
                sample_weight *= k

        drift_input = abs(y - y_pred)
        old_error = self._error_tracker.mean.get()
        # Update the drift detector
        self.drift_detector.update(drift_input)
        self._error_tracker.update(drift_input)
        error_change = self.drift_detector.drift_detected

        # Error is decreasing
        if error_change and self._error_tracker.mean.get() < old_error:
            # Reset the error estimator
            self._error_tracker = self._error_tracker.clone()

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

        self._error_tracker = st.Var()

    def traverse(self, x, until_leaf=True) -> list[HTLeaf]:  # type: ignore
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
        found_nodes: list[HTLeaf] = []
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

        drift_input = abs(y - y_pred)
        old_error = self._error_tracker.mean.get()

        self.drift_detector.update(drift_input)
        self._error_tracker.update(drift_input)
        error_change = self.drift_detector.drift_detected

        # Error is decreasing, better to keep things as they are
        if error_change and self._error_tracker.mean.get() < old_error:
            # Reset the error estimator
            self._error_tracker = self._error_tracker.clone()
            error_change = False

        # Condition to build a new alternate tree
        if error_change and not self._alternate_tree:
            # Reset the error estimator
            self._error_tracker = self._error_tracker.clone()
            self._alternate_tree = tree._new_leaf(parent=self)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1
        # Condition to replace alternate tree
        elif self._alternate_tree:
            alt_n = self._alternate_tree._error_tracker.mean.n
            cur_n = self._error_tracker.mean.n
            # As the size threshold is usually large, we can run a two-tailed z-test to decide whether
            # the alternate subtree has significantly smaller prediction errors than the main subtree
            if alt_n > tree.drift_window_threshold and cur_n > tree.drift_window_threshold:
                # Mean prediction error
                alt_mean_er = self._alternate_tree._error_tracker.mean.get()
                cur_mean_er = self._error_tracker.mean.get()

                # Variance of the prediction error
                alt_s2_er = self._alternate_tree._error_tracker.get()
                cur_s2_er = self._error_tracker.get()

                # Perform z-test to determine whether mean errors are significantly different
                z = (alt_mean_er - cur_mean_er) / math.sqrt(alt_s2_er / alt_n + cur_s2_er / cur_n)
                # We double the p-value due to the double-tailed test
                p_value = 2.0 * tree._norm_dist.cdf(-abs(z))

                # The mean errors are significantly different accordingly to the z-test
                if p_value <= tree.switch_significance:
                    # The alternate tree is the best option
                    if alt_mean_er < cur_mean_er:
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
                    # The current subtree yields the best results
                    else:
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
