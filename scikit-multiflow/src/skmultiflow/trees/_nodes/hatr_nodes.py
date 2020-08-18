import math

from skmultiflow.drift_detection.adwin import ADWIN
from .base import FoundNode
from .base import ActiveLeaf
from .base import InactiveLeaf
from .hatc_nodes import AdaNode
from .hatc_nodes import AdaSplitNode
from .htr_nodes import ActiveLearningNodePerceptron


class AdaSplitNodeRegressor(AdaSplitNode):
    """ Node that splits the data in a Hoeffding Adaptive Tree regressor.

    Parameters
    ----------
    split_test: skmultiflow.split_test.InstanceConditionalTest
        Split test.
    stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, split_test, stats=None, random_state=None):
        super().__init__(split_test, stats, random_state)
        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._n = 0

    # Override AdaSplitNode
    def learn_one(self, X, y, weight, tree, parent, parent_branch):
        normalized_error = 0.0

        leaf = self.filter_instance_to_leaf(X, parent, parent_branch).node
        if leaf is not None:
            y_pred = leaf.predict_one(X, tree=tree)
            normalized_error = get_normalized_error(y, y_pred, self)
        if self._adwin is None:
            self._adwin = ADWIN()

        old_error = self.error_estimation

        # Add element to change detector
        self._adwin.add_element(normalized_error)

        # Detect change
        self.error_change = self._adwin.detected_change()

        if self.error_change and old_error > self.error_estimation:
            self.error_change = False

        # Check condition to build a new alternate tree
        if self.error_change:
            self._alternate_tree = tree._new_learning_node()
            tree.alternate_trees_cnt += 1

        # Condition to replace alternate tree
        elif self._alternate_tree is not None and not self._alternate_tree.error_is_null():
            if self.error_width > tree._ERROR_WIDTH_THRESHOLD \
                    and self._alternate_tree.error_width > tree._ERROR_WIDTH_THRESHOLD:
                old_error_rate = self.error_estimation
                alt_error_rate = self._alternate_tree.error_estimation
                fDelta = .05
                fN = 1.0 / self._alternate_tree.error_width + 1.0 / self.error_width

                sq_term = 2.0 * old_error_rate * (1.0 - old_error_rate) * math.log(2.0 / fDelta) \
                    * fN
                bound = math.sqrt(sq_term) if sq_term > 0 else 0.0

                if bound < (old_error_rate - alt_error_rate):
                    tree._active_leaf_node_cnt -= self.n_leaves
                    tree._active_leaf_node_cnt += self._alternate_tree.n_leaves
                    self.kill_tree_children(tree)

                    if parent is not None:
                        parent.set_child(parent_branch, self._alternate_tree)
                    else:
                        tree._tree_root = tree._tree_root._alternate_tree
                    tree.switch_alternate_trees_cnt += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, ActiveLeaf):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, InactiveLeaf):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_children(tree)
                    tree.pruned_alternate_trees_cnt += 1  # hat.pruned_alternate_trees_cnt to check

        # Learn one sample in alternate tree and child nodes
        if self._alternate_tree is not None:
            self._alternate_tree.learn_one(X, y, weight, tree, parent, parent_branch)
        child_branch = self.instance_child_index(X)
        child = self.get_child(child_branch)

        if child is not None:
            child.learn_one(X, y, weight, tree, parent=self, parent_branch=child_branch)
        # Instance contains a categorical value previously unseen by the split node
        else:
            # Creates a new learning node to encompass the new observed feature
            # value
            leaf_node = tree._new_learning_node()
            branch_id = self.split_test.add_new_branch(
                X[self.split_test.get_atts_test_depends_on()[0]]
            )
            self.set_child(branch_id, leaf_node)
            tree._active_leaf_node_cnt += 1
            leaf_node.learn_one(X, y, weight, tree, parent, parent_branch)

    def predict_one(self, X, *, tree=None):
        # Called in case an emerging categorical feature has no path down the split node to be
        # sorted
        return self.stats[1] / self.stats[0] if len(self.stats) > 0 else 0.0

    # override AdaNode
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts=False, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        if update_splitter_counts:
            try:
                self._stats[0] += weight
                self._stats[1] += y * weight
                self._stats[2] += y * y * weight
            except KeyError:
                self._stats[0] = weight
                self._stats[1] = y * weight
                self._stats[2] = y * y * weight

        child_index = self.instance_child_index(X)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                child.filter_instance_to_leaves(X, y, weight, parent, parent_branch,
                                                update_splitter_counts, found_nodes)
            else:
                found_nodes.append(FoundNode(None, self, child_index))
        if self._alternate_tree is not None:
            self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999,
                                                           update_splitter_counts, found_nodes)


class AdaActiveLearningNodeRegressor(ActiveLearningNodePerceptron, AdaNode):
    """ Learning Node of the Hoeffding Adaptive Tree regressor.

    Always uses a linear perceptron model to provide predictions.

    Parameters
    ----------
    initial_stats: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    parent_node: AdaLearningNodeForRegression (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats=None, parent_node=None, random_state=None):
        super().__init__(initial_stats, parent_node, random_state)
        self._adwin = ADWIN()
        self._error_change = False

        # Normalization of info monitored by drift detectors (using Welford's algorithm)
        self._n = 0

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

    def kill_tree_children(self, hat):
        pass

    def learn_one(self, X, y, weight, tree, parent, parent_branch):
        y_pred = self.predict_one(X, tree=tree)
        normalized_error = get_normalized_error(y, y_pred, self)

        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = self._random_state.poisson(1.0)
            if k > 0:
                weight = weight * k

        if self._adwin is None:
            self._adwin = ADWIN()

        old_error = self.error_estimation

        # Add element to Adwin
        self._adwin.add_element(normalized_error)
        # Detect change with Adwin
        self._error_change = self._adwin.detected_change()

        if self._error_change and old_error > self.error_estimation:
            self._error_change = False

        # Update statistics
        super().learn_one(X, y, weight=weight, tree=tree)

        weight_seen = self.total_weight

        if weight_seen - self.last_split_attempt_at >= tree.grace_period:
            tree._attempt_to_split(self, parent, parent_branch)
            self.last_split_attempt_at = weight_seen

    def predict_one(self, X, *, tree=None):
        prediction_option = tree.leaf_prediction
        if prediction_option == tree._TARGET_MEAN:
            return self._stats[1] / self._stats[0] if len(self._stats) > 0 and self._stats[0] > 0 \
                else 0.0
        else:
            return super().predict_one(X, tree=tree)

    # New for option votes
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        found_nodes.append(FoundNode(self, parent, parent_branch))


def get_normalized_error(y_true, y_pred, node):
    drift_input = abs(y_true - y_pred)

    node._n += 1
    # Welford's algorithm update step
    if node._n == 1:
        node._pM = node._M = drift_input
        node._pS = 0

        return 0.0
    else:
        node._M = node._pM + (drift_input - node._pM) / node._n
        node._S = node._pS + (drift_input - node._pM) * (drift_input - node._M)

        # Save previously calculated values for the next iteration
        node._pM = node._M
        node._pS = node._S

        sd = math.sqrt(node._S / (node._n - 1))

        # Apply z-score normalization to drift input
        norm_input = (drift_input - node._M) / sd if sd > 0 else 0.0

        # Data with zero mean and unit variance -> (empirical rule) 99.73% of the values lie
        # between [mean - 3*sd, mean + 3*sd] (in a normal distribution). We assume this range
        # for the normalized data.
        # Hence, the values are assumed to be between [-3, 3] (as std=1) and we can apply the
        # min-max norm to cope with ADWIN's requirements
        return (norm_input + 3) / 6
