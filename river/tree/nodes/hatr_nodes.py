import math
import typing

from river.drift.adwin import ADWIN
from river.stats import Var
from river.utils.skmultiflow_utils import check_random_state

from .branch import (
    HTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .hatc_nodes import AdaNode
from .htr_nodes import LeafAdaptive, LeafMean, LeafModel
from .leaf import HTLeaf


class AdaLeafRegressor(HTLeaf, AdaNode):
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
    adwin_delta
        The delta parameter of ADWIN.
    seed
        Seed to control the generation of random numbers and support reproducibility.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, adwin_delta, seed, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)

        self.adwin_delta = adwin_delta
        self._adwin = ADWIN(delta=self.adwin_delta)
        self._error_change = False
        self._rng = check_random_state(seed)

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._error_normalizer = Var(ddof=1)

    @property
    def error_estimation(self):
        return self._adwin.estimation

    @property
    def error_width(self):
        return self._adwin.width

    def error_is_null(self):
        return self._adwin is None

    def kill_tree_children(self, hatr):
        pass

    def learn_one(
        self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None
    ):
        y_pred = self.prediction(x, tree=tree)
        normalized_error = normalize_error(y, y_pred, self)

        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = self._rng.poisson(1.0)
            if k > 0:
                sample_weight = sample_weight * k

        if self._adwin is None:
            self._adwin = ADWIN(delta=self.adwin_delta)

        old_error = self.error_estimation

        # Update ADWIN
        self._error_change, _ = self._adwin.update(normalized_error)

        # Error is decreasing
        if self._error_change and old_error > self.error_estimation:
            self._error_change = False

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
                    adwin_delta=tree.adwin_confidence,
                    seed=tree.seed,
                )
                self.last_split_attempt_at = weight_seen


class AdaBranchRegressor(HTBranch, AdaNode):
    """Node that splits the data in a Hoeffding Adaptive Tree Regression.

    Parameters
    ----------
    stats
        Target stats.
    depth
        The depth of the node.
    adwin_delta
        The delta parameter of ADWIN.
    seed
        Internal random state used to sample from poisson distributions.
    attributes
        Other parameters passed to the split node.
    """

    def __init__(self, stats, *children, adwin_delta, seed, **attributes):
        stats = stats if stats else Var()
        super().__init__(stats, *children, **attributes)
        self.adwin_delta = adwin_delta
        self._adwin = ADWIN(delta=self.adwin_delta)
        self._alternate_tree = None
        self._error_change = False

        self._rng = check_random_state(seed)

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._error_normalizer = Var(ddof=1)

    def traverse(self, x, until_leaf=True) -> typing.List[HTLeaf]:
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
        found_nodes = []
        for node in self.walk(x, until_leaf=until_leaf):
            if (
                isinstance(node, AdaBranchRegressor)
                and node._alternate_tree is not None
            ):
                if isinstance(node._alternate_tree, AdaBranchRegressor):
                    found_nodes.append(
                        node._alternate_tree.traverse(x, until_leaf=until_leaf)
                    )
                else:
                    found_nodes.append(node._alternate_tree)

        found_nodes.append(node)
        return found_nodes

    def iter_leaves(self):
        """Iterate over leaves from the left-most one to the right-most one.

        Overrides the base implementation by also including alternate subtrees.
        """
        for child in self.children:
            yield from child.iter_leaves()

            if (
                isinstance(child, AdaBranchRegressor)
                and child._alternate_tree is not None
            ):
                yield from child._alternate_tree.iter_leaves()

    @property
    def error_estimation(self):
        return self._adwin.estimation

    @property
    def error_width(self):
        w = 0.0
        if not self.error_is_null():
            w = self._adwin.width

        return w

    def error_is_null(self):
        return self._adwin is None

    def learn_one(
        self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=None
    ):
        leaf = super().traverse(x, until_leaf=True)
        y_pred = leaf.prediction(x, tree=tree)
        normalized_error = normalize_error(y, y_pred, self)

        # Update stats as traverse the tree to improve predictions (in case split nodes are used
        # to provide responses)
        self.stats.update(y, sample_weight)

        if self._adwin is None:
            self._adwin = ADWIN(self.adwin_delta)

        old_error = self.error_estimation

        # Update ADWIN
        self._error_change, _ = self._adwin.update(normalized_error)

        if self._error_change and old_error > self.error_estimation:
            self._error_change = False

        # Condition to build a new alternate tree
        if self._error_change:
            self._alternate_tree = tree._new_leaf(parent=self)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1

        # Condition to replace alternate tree
        elif (
            self._alternate_tree is not None
            and not self._alternate_tree.error_is_null()
        ):
            if (
                self.error_width > tree.drift_window_threshold
                and self._alternate_tree.error_width > tree.drift_window_threshold
            ):
                old_error_rate = self.error_estimation
                alt_error_rate = self._alternate_tree.error_estimation
                f_delta = 0.05
                f_n = 1.0 / self._alternate_tree.error_width + 1.0 / self.error_width

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
                    if isinstance(self._alternate_tree, HTBranch):
                        self._alternate_tree.kill_tree_children(tree)  # noqa
                    self._alternate_tree = None
                    tree._n_pruned_alternate_trees += 1

        # Learn one sample in alternate tree and child nodes
        if self._alternate_tree is not None:
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
            if isinstance(child, HTBranch):
                if child._alternate_tree is not None:
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
    def __init__(
        self, stats, feature, radius_and_slots, depth, *children, **attributes
    ):
        super().__init__(
            stats, feature, radius_and_slots, depth, *children, **attributes
        )


class AdaLeafRegMean(AdaLeafRegressor, LeafMean):
    def __init__(self, stats, depth, splitter, adwin_delta, seed, **kwargs):
        super().__init__(stats, depth, splitter, adwin_delta, seed, **kwargs)


class AdaLeafRegModel(AdaLeafRegressor, LeafModel):
    def __init__(self, stats, depth, splitter, adwin_delta, seed, **kwargs):
        super().__init__(stats, depth, splitter, adwin_delta, seed, **kwargs)


class AdaLeafRegAdaptive(AdaLeafRegressor, LeafAdaptive):
    def __init__(self, stats, depth, splitter, adwin_delta, seed, **kwargs):
        super().__init__(stats, depth, splitter, adwin_delta, seed, **kwargs)


def normalize_error(y_true, y_pred, node):
    drift_input = y_true - y_pred
    node._error_normalizer.update(drift_input)

    if node._error_normalizer.mean.n == 1:
        return 0.5  # The expected error is the normalized mean error

    sd = math.sqrt(node._error_normalizer.sigma)

    # We assume the error follows a normal distribution -> (empirical rule) 99.73% of the values
    # lie  between [mean - 3*sd, mean + 3*sd]. We assume this range for the normalized data.
    # Hence, we can apply the  min-max norm to cope with  ADWIN's requirements
    return (drift_input + 3 * sd) / (6 * sd) if sd > 0 else 0.5
