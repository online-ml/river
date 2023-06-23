from __future__ import annotations

import math

from river import stats as st
from river.utils.norm import normalize_values_in_dict
from river.utils.random import poisson

from ..utils import do_naive_bayes_prediction
from .branch import (
    DTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .htc_nodes import LeafNaiveBayesAdaptive
from .leaf import HTLeaf


class AdaLeafClassifier(LeafNaiveBayesAdaptive):
    """Learning node for Hoeffding Adaptive Tree.

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
        The detector used internally to monitor drifts.
    rng
        Random number generator used in Poisson sampling.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, drift_detector, rng, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.drift_detector = drift_detector
        self.rng = rng
        self._mean_error = st.Mean()

    def kill_tree_children(self, hat):
        pass

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None):
        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = poisson(rate=1, rng=self.rng)
            if k > 0:
                sample_weight *= k

        aux = self.prediction(x, tree=tree)
        y_pred = max(aux, key=aux.get) if aux else None

        detec_in = 0 if y == y_pred else 1
        old_error = self._mean_error.get()

        # Update the drift detector
        self.drift_detector.update(detec_in)
        self._mean_error.update(detec_in)
        error_change = self.drift_detector.drift_detected

        # Error is decreasing
        if error_change and old_error > self._mean_error.get():
            # Reset the error estimator
            self._mean_error = self._mean_error.clone()

        # Update statistics
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

    # Override LearningNodeNBA
    def prediction(self, x, *, tree=None):
        if not self.stats:
            return

        prediction_option = tree.leaf_prediction
        if not self.is_active() or prediction_option == tree._MAJORITY_CLASS:
            dist = normalize_values_in_dict(self.stats, inplace=False)
        elif prediction_option == tree._NAIVE_BAYES:
            if self.total_weight >= tree.nb_threshold:
                dist = do_naive_bayes_prediction(x, self.stats, self.splitters)
            else:  # Use majority class
                dist = normalize_values_in_dict(self.stats, inplace=False)
        else:  # Naive Bayes Adaptive
            dist = super().prediction(x, tree=tree)

        dist_sum = sum(dist.values())
        curr_error = self._mean_error.get()
        normalization_factor = dist_sum * curr_error * curr_error

        # Weight node's responses accordingly to the estimated error monitored by ADWIN
        # Useful if both the predictions of the alternate tree and the ones from the main tree
        # are combined -> give preference to the most accurate one
        dist = normalize_values_in_dict(dist, normalization_factor, inplace=False)

        return dist


class AdaBranchClassifier(DTBranch):
    """Node that splits the data in a Hoeffding Adaptive Tree.

    Parameters
    ----------
    stats
        Class observations
    adwin_delta
        The delta parameter of ADWIN.
    children
        Sequence of children nodes of this branch.
    attributes
        Other parameters passed to the split node.
    """

    def __init__(self, stats, *children, drift_detector, **attributes):
        super().__init__(stats, *children, **attributes)
        self.drift_detector = drift_detector
        self._alternate_tree = None
        self._mean_error: st.Mean = st.Mean()

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
            if isinstance(node, AdaBranchClassifier) and node._alternate_tree:
                if isinstance(node._alternate_tree, AdaBranchClassifier):
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

            if isinstance(child, AdaBranchClassifier) and child._alternate_tree:
                yield from child._alternate_tree.iter_leaves()

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None):
        leaf = super().traverse(x, until_leaf=True)
        aux = leaf.prediction(x, tree=tree)
        y_pred = max(aux, key=aux.get) if aux else None
        detec_in = 0 if y == y_pred else 1

        # Update stats as traverse the tree to improve predictions (in case split nodes are used
        # to provide responses)
        try:
            self.stats[y] += sample_weight
        except KeyError:
            self.stats[y] = sample_weight

        old_error = self._mean_error.get()

        # Update ADWIN
        self.drift_detector.update(detec_in)
        self._mean_error.update(detec_in)
        error_change = self.drift_detector.drift_detected

        # Classification error is decreasing: skip drift adaptation
        if error_change and old_error > self._mean_error.get():
            # Reset the error estimator
            self._mean_error = self._mean_error.clone()
            error_change = False

        # Condition to build a new alternate tree
        if error_change:
            # Reset the error estimator
            self._mean_error = self._mean_error.clone()
            self._alternate_tree = tree._new_leaf(parent=self)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1
        # Condition to replace alternate tree
        elif self._alternate_tree:
            alt_n_obs = self._alternate_tree._mean_error.n
            n_obs = self._mean_error.n
            if alt_n_obs > tree.drift_window_threshold and n_obs > tree.drift_window_threshold:
                old_error_rate = self._mean_error.get()
                alt_error_rate = self._alternate_tree._mean_error.get()

                n = 1.0 / alt_n_obs + 1.0 / n_obs

                bound = math.sqrt(
                    2.0
                    * old_error_rate
                    * (1.0 - old_error_rate)
                    * math.log(2.0 / tree.switch_significance)
                    * n
                )

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

        if child:
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


class AdaNomBinaryBranchClass(AdaBranchClassifier, NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)


class AdaNumBinaryBranchClass(AdaBranchClassifier, NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)


class AdaNomMultiwayBranchClass(AdaBranchClassifier, NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)


class AdaNumMultiwayBranchClass(AdaBranchClassifier, NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **attributes):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **attributes)
