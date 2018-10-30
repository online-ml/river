import logging
from abc import ABCMeta, abstractmethod
from skmultiflow.utils.utils import get_max_value_key, normalize_values_in_dict
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.utils import do_naive_bayes_prediction
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.utils import check_random_state
import math
import numpy as np

# Logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SplitNode = HoeffdingTree.SplitNode
LearningNodeNBAdaptive = HoeffdingTree.LearningNodeNBAdaptive

MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'
error_width_threshold = 300


class HAT(HoeffdingTree):
    """ Hoeffding Adaptive Tree for evolving data streams.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.

    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.

    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.

    split_criterion: string (default='info_gain')
        Split criterion to use.

        - 'gini' - Gini
        - 'info_gain' - Information Gain

    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.

    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.

    binary_split: boolean (default=False)
        If True, only allow binary splits.

    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.

    no_preprune: boolean (default=False)
        If True, disable pre-pruning.

    leaf_prediction: string (default='nba')
        Prediction mechanism used at leafs.

        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive

    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    The Hoeffding Adaptive Tree [1]_ uses ADWIN [2]_ to monitor performance of branches on the tree and to replace them
    with new branches when their accuracy decreases if the new branches are more accurate.

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavaldà. "Adaptive learning from evolving data streams."\
       In International Symposium on Intelligent Data Analysis, pp. 249-260. Springer, Berlin, Heidelberg, 2009.
    .. [2] Bifet, Albert, and Ricard Gavaldà. "Learning from time-changing data with adaptive windowing."\
       In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448. Society for Industrial\
       and Applied Mathematics, 2007.

    Examples
    --------
    >>> from skmultiflow.trees.hoeffding_adaptive_tree import HAT
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> stream = FileStream("/skmultiflow/data/datasets/covtype.csv", -1, 1)
    >>> stream.prepare_for_use()
    >>>
    >>> classifier = HAT()
    >>> evaluator = EvaluatePrequential(pretrain_size=200, max_samples=50000, batch_size=1, n_wait=200, max_time=1000,
    >>>                                 output_file=None, show_plot=True, metrics=['kappa', 'kappa_t', 'performance'])
    >>>
    >>> evaluator.evaluate(stream=stream, model=classifier)

    """

    class NewNode(metaclass=ABCMeta):
        """
            Abstract Class to create a New Node for HoeffdingAdaptiveTree (HAT)
        """

        @abstractmethod
        def number_leaves(self):
            pass

        @abstractmethod
        def get_error_estimation(self):
            pass

        @abstractmethod
        def get_error_width(self):
            pass

        @abstractmethod
        def is_null_error(self):
            pass

        @abstractmethod
        def kill_tree_children(self, hat):
            pass

        @abstractmethod
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
            pass

        @abstractmethod
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counts, found_nodes=None):
            pass

    class AdaSplitNode(SplitNode, NewNode):
        def __init__(self, split_test, class_observations):
            super().__init__(split_test, class_observations)
            self._estimation_error_weight = ADWIN()
            self._alternate_tree = None
            self.error_change = False
            self._random_seed = 1
            self._classifier_random = check_random_state(self._random_seed)

        # Override NewNode
        def number_leaves(self):
            num_of_leaves = 0
            for child in self._children:
                if child is not None:
                    num_of_leaves += child.number_leaves()

            return num_of_leaves

        # Override NewNode
        def get_error_estimation(self):
            return self._estimation_error_weight.estimation

        # Override NewNode
        def get_error_width(self):
            w = 0.0
            if self.is_null_error() is False:
                w = self._estimation_error_weight.width

            return w

        # Override NewNode
        def is_null_error(self):
            return self._estimation_error_weight is None

        # Override NewNode
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
                if self.get_error_width() > error_width_threshold \
                        and self._alternate_tree.get_error_width() > error_width_threshold:
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
                        if isinstance(self._alternate_tree, HAT.ActiveLearningNode):
                            self._alternate_tree = None
                        elif isinstance(self._alternate_tree, HAT.InactiveLearningNode):
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

        # Override NewNode
        def kill_tree_children(self, hat):
            for child in self._children:
                if child is not None:
                    # Delete alternate tree if it exists
                    if isinstance(child, HAT.AdaSplitNode) and child._alternate_tree is not None:
                        child._alternate_tree.kill_tree_children(hat)
                        self._pruned_alternate_trees += 1
                    # Recursive delete of SplitNodes
                    if isinstance(child, HAT.AdaSplitNode):
                        child.kill_tree_children(hat)

                    if isinstance(child, HAT.ActiveLearningNode):
                        child = None
                        hat._active_leaf_node_cnt -= 1
                    elif isinstance(child, HAT.InactiveLearningNode):
                        child = None
                        hat._inactive_leaf_node_cnt -= 1

        # override NewNode
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
                    found_nodes.append(HoeffdingTree.FoundNode(None, self, child_index))
            if self._alternate_tree is not None:
                self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999,
                                                               update_splitter_counts, found_nodes)

    class AdaLearningNode(LearningNodeNBAdaptive, NewNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._estimation_error_weight = ADWIN()
            self.error_change = False
            self._randomSeed = 1
            self._classifier_random = check_random_state(self._randomSeed)

        # Override NewNode
        def number_leaves(self):
            return 1

        # Override NewNode
        def get_error_estimation(self):
            return self._estimation_error_weight.estimation

        # Override NewNode
        def get_error_width(self):
            return self._estimation_error_weight.width

        # Override NewNode
        def is_null_error(self):
            return self._estimation_error_weight is None

        def kill_tree_children(self, hat):
            pass

        # Override NewNode
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
            true_class = y

            # k = self._classifier_random.poisson(1.0)
            # if k > 0:
            #     weight = weight * k

            tmp = self.get_class_votes(X, hat)

            class_prediction = get_max_value_key(tmp)

            bl_correct = (true_class == class_prediction)

            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()

            # Add element to Adwin
            add = 0.0 if (bl_correct is True) else 1.0

            self._estimation_error_weight.add_element(add)
            # Detect change with Adwin
            self.error_change = self._estimation_error_weight.detected_change()

            if self.error_change is True and old_error > self.get_error_estimation():
                self.error_change = False

            # Update statistics
            super().learn_from_instance(X, y, weight, hat)

            # call ActiveLearningNode
            weight_seen = self.get_weight_seen()

            if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= hat.grace_period:
                hat._attempt_to_split(self, parent, parent_branch)
                self.set_weight_seen_at_last_split_evaluation(weight_seen)

        # Override LearningNodeNBAdaptive
        def get_class_votes(self, X, ht):
            # dist = {}
            prediction_option = ht.leaf_prediction
            # MC
            if prediction_option == MAJORITY_CLASS:
                dist = self.get_observed_class_distribution()
            # NB
            elif prediction_option == NAIVE_BAYES:
                dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            # NBAdaptive
            else:
                if self._mc_correct_weight > self._nb_correct_weight:
                    dist = self.get_observed_class_distribution()
                else:
                    dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

            dist_sum = sum(dist.values())  # sum all values in dictionary
            normalization_factor = dist_sum * self.get_error_estimation() * self.get_error_estimation()

            if normalization_factor > 0.0:
                normalize_values_in_dict(dist, normalization_factor)

            return dist

        # Override NewNode, New for option votes
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counts, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
            found_nodes.append(HoeffdingTree.FoundNode(self, parent, parent_branch))

    # =============================================
    # == Hoeffding Adaptive Tree implementation ===
    # =============================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None):

        super(HAT, self).__init__(max_byte_size=max_byte_size,
                                  memory_estimate_period=memory_estimate_period,
                                  grace_period=grace_period,
                                  split_criterion=split_criterion,
                                  split_confidence=split_confidence,
                                  tie_threshold=tie_threshold,
                                  binary_split=binary_split,
                                  stop_mem_management=stop_mem_management,
                                  remove_poor_atts=remove_poor_atts,
                                  no_preprune=no_preprune,
                                  leaf_prediction=leaf_prediction,
                                  nb_threshold=nb_threshold,
                                  nominal_attributes=nominal_attributes)
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self._tree_root = None

    def reset(self):
        self.alternate_trees_cnt = 0
        self.pruned_alternate_trees_cnt = 0
        self.switch_alternate_trees_cnt = 0
        self._tree_root = None

    # Override HoeffdingTree
    def _partial_fit(self, X, y, weight):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        if isinstance(self._tree_root, self.InactiveLearningNode):
            self._tree_root.learn_from_instance(X, y, weight, self)
        else:
            self._tree_root.learn_from_instance(X, y, weight, self, None, -1)

    def filter_instance_to_leaves(self, X, y, weight, split_parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, y, weight, split_parent, parent_branch,
                                                  update_splitter_counts, nodes)
        return nodes

    # Override HoeffdingTree
    def get_votes_for_instance(self, X):
        result = {}
        if self._tree_root is not None:
            if isinstance(self._tree_root, self.InactiveLearningNode):
                found_node = [self._tree_root.filter_instance_to_leaf(X, None, -1)]
            else:
                found_node = self.filter_instance_to_leaves(X, -np.inf, -np.inf, None, -1, False)
            for fn in found_node:
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)  # add elements to dictionary
        return result

    def score(self, X, y):
        raise NotImplementedError

    # Override HoeffdingTree
    def _new_learning_node(self, initial_class_observations=None):
        return self.AdaLearningNode(initial_class_observations)

    # Override HoeffdingTree
    def new_split_node(self, split_test, class_observations):
        return self.AdaSplitNode(split_test, class_observations)
