import math

from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees.nodes import FoundNode
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import InactiveLearningNode
from skmultiflow.trees.nodes import AdaNode
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import check_random_state, get_max_value_key

ERROR_WIDTH_THRESHOLD = 300


class AdaSplitNode(SplitNode, AdaNode):
    """ Node that splits the data in a Hoeffding Adaptive Tree.

    Parameters
    ----------
    split_test: skmultiflow.split_test.InstanceConditionalTest
        Split test.
    class_observations: dict (class_value, weight) or None
        Class observations
    """
    def __init__(self, split_test, class_observations):
        super().__init__(split_test, class_observations)
        self._estimation_error_weight = ADWIN()
        self._alternate_tree = None
        self.error_change = False
        self._random_seed = 1
        self._classifier_random = check_random_state(self._random_seed)

    # Override AdaNode
    def number_leaves(self):
        num_of_leaves = 0
        for child in self._children:
            if child is not None:
                num_of_leaves += child.number_leaves()

        return num_of_leaves

    # Override AdaNode
    def get_error_estimation(self):
        return self._estimation_error_weight.estimation

    # Override AdaNode
    def get_error_width(self):
        w = 0.0
        if self.is_null_error() is False:
            w = self._estimation_error_weight.width

        return w

    # Override AdaNode
    def is_null_error(self):
        return self._estimation_error_weight is None

    # Override AdaNode
    def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
        true_class = y
        class_prediction = 0

        leaf = self.filter_instance_to_leaf(X, parent, parent_branch)
        if leaf.node is not None:
            class_prediction = get_max_value_key(leaf.node.get_class_votes(X, hat))

        bl_correct = (true_class == class_prediction)

        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to ADWIN
        add = 0.0 if (bl_correct is True) else 1.0

        self._estimation_error_weight.add_element(add)
        # Detect change with ADWIN
        self.error_change = self._estimation_error_weight.detected_change()

        if self.error_change is True and old_error > self.get_error_estimation():
            self.error_change = False

        # Check condition to build a new alternate tree
        if self.error_change is True:
            self._alternate_tree = hat._new_learning_node()
            hat.alternate_trees_cnt += 1

        # Condition to replace alternate tree
        elif self._alternate_tree is not None and self._alternate_tree.is_null_error() is False:
            if self.get_error_width() > ERROR_WIDTH_THRESHOLD \
                    and self._alternate_tree.get_error_width() > ERROR_WIDTH_THRESHOLD:
                old_error_rate = self.get_error_estimation()
                alt_error_rate = self._alternate_tree.get_error_estimation()
                fDelta = .05
                fN = 1.0 / self._alternate_tree.get_error_width() + 1.0 / (self.get_error_width())

                bound = math.sqrt(2.0 * old_error_rate * (1.0 - old_error_rate) * math.log(2.0 / fDelta) * fN)
                # To check, bound never less than (old_error_rate - alt_error_rate)
                if bound < (old_error_rate - alt_error_rate):
                    hat._active_leaf_node_cnt -= self.number_leaves()
                    hat._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                    self.kill_tree_children(hat)

                    if parent is not None:
                        parent.set_child(parent_branch, self._alternate_tree)
                    else:
                        # Switch tree root
                        hat._tree_root = hat._tree_root.alternateTree
                    hat.switch_alternate_trees_cnt += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, ActiveLearningNode):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, InactiveLearningNode):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_children(hat)
                    hat.pruned_alternate_trees_cnt += 1  # hat.pruned_alternate_trees_cnt to check

        # Learn_From_Instance alternate Tree and Child nodes
        if self._alternate_tree is not None:
            self._alternate_tree.learn_from_instance(X, y, weight, hat, parent, parent_branch)
        child_branch = self.instance_child_index(X)
        child = self.get_child(child_branch)
        if child is not None:
            child.learn_from_instance(X, y, weight, hat, parent, parent_branch)

    # Override AdaNode
    def kill_tree_children(self, hat):
        for child in self._children:
            if child is not None:
                # Delete alternate tree if it exists
                if isinstance(child, SplitNode) and child._alternate_tree is not None:
                    child._alternate_tree.kill_tree_children(hat)
                    self._pruned_alternate_trees += 1
                # Recursive delete of SplitNodes
                if isinstance(child, SplitNode):
                    child.kill_tree_children(hat)

                if isinstance(child, ActiveLearningNode):
                    child = None
                    hat._active_leaf_node_cnt -= 1
                elif isinstance(child, InactiveLearningNode):
                    child = None
                    hat._inactive_leaf_node_cnt -= 1

    # override AdaNode
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts=False, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        if update_splitter_counts:
            try:
                self._observed_class_distribution[y] += weight  # Dictionary (class_value, weight)
            except KeyError:
                self._observed_class_distribution[y] = weight
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
