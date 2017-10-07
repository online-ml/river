__author__ = 'Jacob Montiel'

import sys
import numpy as np
from abc import ABCMeta
from skmultiflow.classification.base import BaseClassifier
from skmultiflow.classification.core.attribute_class_observers.gaussian_numeric_attribute_class_observer\
    import GaussianNumericAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.nominal_attribute_class_observer\
    import NominalAttributeClassObserver
from skmultiflow.classification.core.attribute_class_observers.null_attribute_class_observer\
    import NullAttributeClassObserver
from skmultiflow.classification.core.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.classification.core.split_criteria.gini_split_criterion import GiniSplitCriterion
from skmultiflow.classification.core.split_criteria.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction

class HoeffdingTree(BaseClassifier):
    '''
    Hoeffding Tree or VFDT.

    A Hoeffding tree is an incremental, anytime decision tree induction algorithm that is capable of learning from
    massive data streams, assuming that the distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an optimal splitting attribute. This idea is
    supported mathematically by the Hoeffding bound, which quantiﬁes the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision tree learners is that
    it has sound guarantees of performance. Using the Hoeffding bound one can show that its output is asymptotically
     nearly identical to that of a non-incremental learner using inﬁnitely many examples.

     See for details:
     G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
    In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    Implementation based on MOA:
    Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer (2010);
    MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604

    Parameters:
    -----------

    '''

    class FoundNode(object):
        def __init__(self, node=None, parent=None, parent_branch=None):
            self.node = node
            self.parent = parent
            self.parent_branch = parent_branch

    class Node(metaclass=ABCMeta):
        """
        Base class for Nodes in a Hoeffding Tree
        """

        def __init__(self, class_observations=None):
            if class_observations is None:
                class_observations = {}   # Dictionary (class_value, weight)
            self._observed_class_distribution = class_observations

        def is_leaf(self):
            return True

        def filter_instance_to_leaf(self, X, y, parent, parent_branch):
            return HoeffdingTree.FoundNode(self, parent, parent_branch)

        def get_observed_class_distribution(self):
            return self._observed_class_distribution

        def get_class_votes(self, X, y, ht):
            return self._observed_class_distribution

        def observed_class_distribution_is_pure(self):
            count = 0
            for _, weight in self._observed_class_distribution.items():
                if weight is not 0:
                    count += 1
                    if count > 1:   # No need to count beyond this point
                        break
            return count < 2

        def subtree_depth(self):
            return 0

        def calculate_promise(self):
            total_seen = sum(self._observed_class_distribution.values())
            if total_seen > 0:
                return total_seen - max(self._observed_class_distribution.values())
            else:
                return 0

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._observed_class_distribution)

        def calc_byte_size_including_subtree(self):
            return self.__sizeof__()

        # TODO
        def describe_subtree(self):
            pass

        def get_description(self):
            pass


    class SplitNode(Node):
        """
        Class for a node that splits the data in a Hoeffding tree
        """

        def __init__(self, split_test, class_observations, size=-1):
            super().__init__(class_observations)
            self._split_test = split_test
            # Dict of tuples (branch, child)
            if size > 0:
                self._children = []*size
            else:
                self._children = []

        def num_children(self):
            return len(self._children)

        def set_child(self, index, node):
            self._children[index] = node

        def get_child(self, index):
            return self._children[index]

        def instance_child_index(self, X, y):
            return self._split_test.branch_for_instance(X, y)

        def is_leaf(self):
            return False

        def filter_instance_to_leaf(self, X, y, parent, parent_branch):
            child_index = self.instance_child_index(X, y);
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    return child.filter_instance_to_leaf(X, y, self, child_index)
                else:
                    return HoeffdingTree.FoundNode(None, self, child_index)
            else:
                return HoeffdingTree.FoundNode(self, parent, parent_branch)

        def subtree_depth(self):
            max_child_depth = 0
            for child in self._children:
                if child is not None:
                    depth = child.subtree_depth()
                    if depth > max_child_depth:
                        max_child_depth = depth
            return max_child_depth + 1

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._children) + sys.getsizeof(self._split_test)

        def calc_byte_size_including_subtree(self):
            byte_size = self.__sizeof__()
            for child in self._children:
                if child is not None:
                    byte_size += child.calc_byte_size_including_subtree()
            return byte_size

        # TODO
        def describe_subtree(self):
            pass


    class LearningNode(Node):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            pass


    class InactiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            if y > len(self._observed_class_distribution) -1:
                return
            self._observed_class_distribution[y] += weight


    class ActiveLearningNode(LearningNode):
        """A Hoeffding Tree node that supports growth."""
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
            self._is_initialized = False
            self._attribute_observers = []

        def learn_from_instance(self, X, y, weight, ht):
            """ learn_from_instance
            Update the node with the supplied instance.

            Parameters
            ----------
            X: The attributes for updating the node
            y: The class
            weight: The instance's weight
            ht: The Hoeffding Tree

            """
            if not self._is_initialized:
                self._attribute_observers = [None]*len(X)
                self._is_initialized = True
            self._observed_class_distribution[y] += weight

            for i in range(len(X)):
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:    # TODO define
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            return sum(self._observed_class_distribution.values())

        def get_weight_seen_at_last_split_evaluation(self):
            return self._weight_seen_at_last_split_evaluation

        def set_weight_seen_at_last_split_evaluation(self, weight):
            self._weight_seen_at_last_split_evaluation = weight

        def get_best_split_suggestions(self, criterion, ht):
            """ get_best_split_suggestions

            Return a list of the possible split candidates.

            Parameters
            ----------
            criterion: The splitting criterion to be used.
            ht: The Hoeffding Tree

            Returns
            -------
            A list of the possible split candidates.

            """
            best_suggestions = []
            pre_split_dist = [self._observed_class_distribution]
            if not ht.no_pre_prune_option:
                # Add null split as an option
                null_split = AttributeSplitSuggestion(None, pre_split_dist,
                                                      criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))  # TODO check
                best_suggestions.append(null_split)
            for i, obs in enumerate(self._attribute_observers):
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, ht.binary_split_option)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
            return best_suggestions

        def disable_attribute(self, att_idx):
            if att_idx < len(self._attribute_observers) and att_idx > 0:
                self._attribute_observers[att_idx] = NullAttributeClassObserver()

    class LearningNodeNB(ActiveLearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, y, ht):
            if self.get_weight_seen() >= ht._nb_threshold_option:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, y, ht)

        def disable_attribute(self, att_index):
            # Should not disable poor attributes, they are used in NB calculation
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            if max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight,ht)

    def __init__(self):
        # DEFAULT OPTIONS
        self.max_byte_size_option = 33554432                                      # Maximum memory consumed by the tree.
        self.numeric_estimator_option = 'GaussianNumericAttributeClassObserver'              # Numeric estimator to use.
        self.numeric_estimator_option = 'NominalAttributeClassObserver'                      # Nominal estimator to use.
        self.memory_estimate_period_option = 1000000             # How many instances between memory consumption checks.
        self.grace_period_option = 200           # The number of instances a leaf should observe between split attempts.
        self.GINI_SPLIT = 0
        self.INFO_GAIN_SPLIT = 1
        self.split_criterion_option = self.INFO_GAIN_SPLIT                                     # Split criterion to use.
        self.split_confidence_option = 0.0000001  # Allowed error in split decision, closer to 0 takes longer to decide.
        self.tie_threshold_option = 0.05                   # Threshold below which a split will be forced to break ties.
        self.binary_split_option = False                                             # If True only allow binary splits.
        self.stop_mem_management_option = False                   # If True stop growing as soon as memory limit is hit.
        self.remove_poor_atts_option = False                                          # If True disable poor attributes.
        self.no_pre_prune_option = False                                                  # If True disable pre-pruning.

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self.MAJORITY_CLASS = 0
        self.NAIVE_BAYES = 1
        self.NAIVE_BAYES_ADAPTIVE = 2
        self._leaf_prediction_option = self.MAJORITY_CLASS
        self._nb_threshold_option = 0     # The number of instances a leaf should observe before permitting Naive Bayes.

    def __sizeof__(self):
        size = object.__sizeof__(self)
        if self._tree_root is not None:
            size += self._tree_root.calc_byte_size_including_subtree()
        return size

    def measure_byte_size(self):
        return self.__sizeof__()

    def reset_learning_imp(self):
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        if self._leaf_prediction_option != self.MAJORITY_CLASS:
            self.remove_poor_atts_option = None


    def partial_fit(self, X, y, weight, classes=None):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        if isinstance(self._tree_root, self.LearningNode):
            found_node = self._tree_root.filter_instance_to_leaf(X, y, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = self._new_learning_node()
                found_node.parent.set_child(found_node.parent_branch, leaf_node)
                self._active_leaf_node_cnt += 1
            if isinstance(leaf_node, self.LearningNode):
                learning_node = leaf_node
                learning_node.learn_from_instance(X, y, weight, self)
                if self._growth_allowed and isinstance(learning_node, self.ActiveLearningNode):
                    active_learning_node = learning_node
                    weight_seen = active_learning_node.get_weight_seen()
                    weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                    if weight_diff >= self.grace_period_option:
                        self.attempt_to_split(active_learning_node, found_node.parent, found_node.parent_branch)
                        active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
            if self.train_weight_seen_by_model % self.memory_estimate_period_option == 0:
                self.estimate_model_byte_size()

    def get_votes_for_instance(self, X, y, weight):                                       # TODO do we need this?
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, y, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.get_class_votes(X, y, weight, self)
        else:
            return {}

    def get_model_measuraments(self):
        measurements = {}
        measurements['Tree size (nodes)'] = self._decision_node_cnt +\
                                            self._active_leaf_node_cnt +\
                                            self._inactive_leaf_node_cnt
        measurements['Tree size (leaves)'] = self._active_leaf_node_cnt + self._inactive_leaf_node_cnt
        measurements['Active learning nodes'] = self._active_leaf_node_cnt
        measurements['Tree depth'] = self.measure_tree_depth()
        measurements['Active leaf byte size estimate'] = self._active_leaf_byte_size_estimate
        measurements['Inctive leaf byte size estimate'] = self._inactive_leaf_byte_size_estimate
        measurements['Byte size estimate overhead'] = self._byte_size_estimate_overhead_fraction

    def measure_tree_depth(self):
        if isinstance(self._tree_root, self.Node):
            return self._tree_root.subtree_depth()
        return 0

    def _new_learning_node(self, initial_class_observations={}):
        if self._leaf_prediction_option == self.MAJORITY_CLASS:
            return self.ActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction_option == self.NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations)

    def get_model_description(self):
        pass    # TODO

    def is_randomizable(self):    # TODO do we need this?
        return False

    def compute_hoeffding_bound(self, range_val, confidence, n):
        return np.sqrt( (range_val * range_val * np.log(1.0/confidence)) / (2.0 * n))

    def new_split_node(self, split_test, class_observations, size=-1):
        return self.SplitNode(split_test, class_observations, size)

    def attempt_to_split(self, node:ActiveLearningNode, parent:SplitNode, parent_idx:int):
        if not node.observed_class_distribution_is_pure():
            if self.split_criterion_option == self.GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self.split_criterion_option == self.INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(
                                       split_criterion.get_range_of_merit(node.get_observed_class_distribution()))
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound or \
                    hoeffding_bound < self.tie_threshold_option:
                    should_split = True
                if self.remove_poor_atts_option is not None and self.remove_poor_atts_option:
                    poor_atts = set()
                    # Scan 1 - add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit > hoeffding_bound:
                                    poor_atts.add(int(split_atts[0]))
                    # Scan 2 - remove good attributes from set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit < hoeffding_bound:
                                    poor_atts.remove(int(split_atts[0]))
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision == None or not isinstance(split_decision, AttributeSplitSuggestion):
                    # Preprune - null wins
                    self.deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self.new_split_node(split_decision.split_test,
                                                    node.get_observed_class_distribution(),
                                                    split_decision.num_splits())
                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(split_decision.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += split_decision.num_splits()
                    if parent is None:
                        self._tree_root = new_split
                    else:
                        parent.set_child(parent_idx, new_split)
                # Manage memory
                self.enforce_tracker_limit()

    def enforce_tracker_limit(self):
        byte_size = (self._active_leaf_byte_size_estimate
                     + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate)\
                    * self._byte_size_estimate_overhead_fraction
        if self._inactive_leaf_node_cnt > 0 or byte_size > self.max_byte_size_option:
            self._growth_allowed = False
            return
        learning_nodes = self.find_learning_nodes()
        learning_nodes.sort(key=lambda n:n.node.calculate_promise())
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if ((max_active * self._active_leaf_byte_size_estimate + (len(learning_nodes) - max_active)
                * self._inactive_leaf_byte_size_estimate) * self._byte_size_estimate_overhead_fraction) \
                > self.max_byte_size_option:
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if isinstance(learning_nodes[i].node, self.ActiveLearningNode):
                self.deactivate_learning_node(learning_nodes[i].node,
                                              learning_nodes[i].parent,
                                              learning_nodes[i].parent_branch)
        for i in range(cutoff, len(learning_nodes)):
            if isinstance(learning_nodes[i].node, self.InactiveLearningNode):
                self.activate_learning_node(learning_nodes[i].node,
                                            learning_nodes[i].parent,
                                            learning_nodes[i].parent_branch)

    def estimate_model_byte_size(self):
        learning_nodes = self.find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if isinstance(found_node, self.ActiveLearningNode):
                total_active_size += sys.getsizeof(found_node.node)
            else:
                total_inactive_size +=  sys.getsizeof(found_node.node)
        if total_active_size > 0:
            self._active_leaf_byte_size_estimate = total_active_size / self._active_leaf_node_cnt
        if total_inactive_size > 0:
            self._inactive_leaf_byte_size_estimate = total_inactive_size / self._inactive_leaf_node_cnt
        actual_model_size = self.measure_byte_size()
        estimated_model_size = (self._active_leaf_node_cnt * self._active_leaf_byte_size_estimate
                                + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate)
        self._byte_size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self.max_byte_size_option:
            self.enforce_tracker_limit()

    def deactivate_all_leaves(self):    # TODO do we need this?
        learning_nodes = self.find_learning_nodes()
        for i in range(len(learning_nodes)):
            if isinstance(learning_nodes[i], self.ActiveLearningNode):
                self.deactivate_learning_node(learning_nodes[i].node,
                                              learning_nodes[i].parent,
                                              learning_nodes[i].parent_branch)

    def deactivate_learning_node(self, to_deactivate:ActiveLearningNode, parent:SplitNode, parent_branch:int):
        new_leaf = self.InactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def activate_learning_node(self, to_activate:InactiveLearningNode, parent:SplitNode, parent_branch:int):
        new_leaf = self._new_learning_node(to_activate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def find_learning_nodes(self):
        found_list = []
        self._find_learning_nodes(self._tree_root, None, -1, found_list)
        return found_list

    def _find_learning_nodes(self, node, parent, parent_branch, found):
        if node is not None:
            if isinstance(node, self.LearningNode):
                found.append(self.FoundNode(node, parent, parent_branch))
            if isinstance(node, self.SplitNode):
                split_node = node
                for i in range(split_node.num_children()):
                    self._find_learning_nodes(split_node.get_child(i), split_node, i, found)

