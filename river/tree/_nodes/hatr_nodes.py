import math

from river.stats import Var
from river.drift.adwin import ADWIN
from river.utils.skmultiflow_utils import check_random_state

from .base import FoundNode
from .base import SplitNode
from .hatc_nodes import AdaNode
from .htr_nodes import LearningNodeAdaptive


class AdaLearningNodeRegressor(LearningNodeAdaptive, AdaNode):
    """Learning Node of the Hoeffding Adaptive Tree regressor.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the learning node in the tree.
    attr_obs
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attr_obs_params
        The parameters passed to the numeric attribute observer algorithm.
    adwin_delta
        The delta parameter of ADWIN.
    seed
        Seed to control the generation of random numbers and support reproducibility.
    """

    def __init__(self, stats, depth, attr_obs, attr_obs_params, leaf_model, adwin_delta, seed):
        super().__init__(stats, depth, attr_obs, attr_obs_params, leaf_model)

        self.adwin_delta = adwin_delta
        self._adwin = ADWIN(delta=self.adwin_delta)
        self.error_change = False
        self._rng = check_random_state(seed)

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._error_normalizer = Var(ddof=1)

    @property
    def n_leaves(self):
        return 1

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

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, parent=None, parent_branch=-1):
        y_pred = self.leaf_prediction(x, tree=tree)
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
            else:
                tree._attempt_to_split(self, parent, parent_branch)
                self.last_split_attempt_at = weight_seen

    def leaf_prediction(self, x, *, tree=None):
        prediction_option = tree.leaf_prediction
        if prediction_option == tree._TARGET_MEAN:
            return self.stats.mean.get()
        elif prediction_option == tree._MODEL:
            return self._leaf_model.predict_one(x)
        else:  # adaptive node
            return super().leaf_prediction(x, tree=tree)

    # Override AdaNode: enable option vote (query potentially more than one leaf for responses)
    def filter_instance_to_leaves(self, x, parent, parent_branch, found_nodes):
        found_nodes.append(FoundNode(self, parent, parent_branch))


class AdaSplitNodeRegressor(SplitNode, AdaNode):
    """Node that splits the data in a Hoeffding Adaptive Tree Regression.

    Parameters
    ----------
    split_test
        Split test.
    stats
        Target stats.
    depth
        The depth of the node.
    adwin_delta
        The delta parameter of ADWIN.
    seed
        Internal random state used to sample from poisson distributions.
    """

    def __init__(self, split_test, stats, depth, adwin_delta, seed):
        stats = stats if stats else Var()
        super().__init__(split_test, stats, depth)
        self.adwin_delta = adwin_delta
        self._adwin = ADWIN(delta=self.adwin_delta)
        self._alternate_tree = None
        self._error_change = False

        self._rng = check_random_state(seed)

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._error_normalizer = Var(ddof=1)

    @property
    def n_leaves(self):
        num_of_leaves = 0
        for child in self._children.values():
            if child is not None:
                num_of_leaves += child.n_leaves

        return num_of_leaves

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

    # Override AdaSplitNodeClassifier
    def learn_one(self, x, y, sample_weight, tree, parent, parent_branch):
        normalized_error = 0.0

        leaf = self.filter_instance_to_leaf(x, parent, parent_branch).node
        if leaf is not None:
            y_pred = leaf.leaf_prediction(x, tree=tree)
        else:
            y_pred = parent.leaf_prediction(x, tree=tree)

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
            self._alternate_tree = tree._new_learning_node(parent=self)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1

        # Condition to replace alternate tree
        elif self._alternate_tree is not None and not self._alternate_tree.error_is_null():
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
                        parent.set_child(parent_branch, self._alternate_tree)
                        self._alternate_tree = None
                    else:
                        # Switch tree root
                        tree._tree_root = tree._tree_root._alternate_tree
                    tree._n_switch_alternate_trees += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, SplitNode):
                        self._alternate_tree.kill_tree_children(tree)
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
        child_branch = self.instance_child_index(x)
        child = self.get_child(child_branch)
        if child is not None:
            child.learn_one(
                x,
                y,
                sample_weight=sample_weight,
                tree=tree,
                parent=self,
                parent_branch=child_branch,
            )
        elif self.split_test.branch_for_instance(x) == -1:
            split_feat = self.split_test.attrs_test_depends_on()[0]
            # Instance contains a categorical value previously unseen by the split node
            if self.split_test.max_branches() == -1 and split_feat in x:
                # Creates a new learning node to encompass the new observed feature value
                leaf_node = tree._new_learning_node(parent=self)
                branch_id = self.split_test.add_new_branch(x[split_feat])
                self.set_child(branch_id, leaf_node)
                tree._n_active_leaves += 1
                leaf_node.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=tree,
                    parent=self,
                    parent_branch=branch_id,
                )
            # The split feature is missing in the instance. Hence, we pass the new example
            # to the most traversed path in the current subtree
            else:
                path = max(
                    self._children,
                    key=lambda c: self._children[c].total_weight if self._children[c] else 0.0,
                )
                leaf_node = self.get_child(path)
                # Pass instance to the most traversed path
                if leaf_node is None:
                    leaf_node = tree._new_learning_node(parent=self)
                    self.set_child(path, leaf_node)
                    tree._n_active_leaves += 1

                leaf_node.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=tree,
                    parent=self,
                    parent_branch=path,
                )

    def leaf_prediction(self, x, *, tree=None):
        # Called in case an emerging categorical feature has no path down the split node to be
        # sorted
        return self.stats.mean.get()

    # Override AdaNode
    def kill_tree_children(self, tree):
        for child_id, child in self._children.items():
            if child is not None:
                # Delete alternate tree if it exists
                if not child.is_leaf():
                    if child._alternate_tree is not None:
                        child._alternate_tree.kill_tree_children(tree)
                        tree._n_pruned_alternate_trees += 1
                        child._alternate_tree = None

                    # Recursive delete of SplitNodes
                    child.kill_tree_children(tree)
                    tree._n_decision_nodes -= 1
                else:
                    if child.is_active():
                        tree._n_active_leaves -= 1
                    else:
                        tree._n_inactive_leaves -= 1

                self._children[child_id] = None

    # override AdaNode
    def filter_instance_to_leaves(self, x, parent, parent_branch, found_nodes):
        child_index = self.instance_child_index(x)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                child.filter_instance_to_leaves(x, parent, parent_branch, found_nodes)
            else:
                found_nodes.append(FoundNode(None, self, child_index))
        else:
            # Emerging value in a categorical feature appears or the split feature is missing from
            # the instance: use parent node in both cases
            found_nodes.append(FoundNode(None, self, child_index))

        if self._alternate_tree is not None:
            self._alternate_tree.filter_instance_to_leaves(x, self, -999, found_nodes)


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
