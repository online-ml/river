__author__ = 'Alessandro Longobardi'

import logging
import random
from abc import ABCMeta, abstractmethod
from skmultiflow.core.utils.utils import *
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
from skmultiflow.classification.core.utils.utils import do_naive_bayes_prediction
from skmultiflow.classification.core.utils.utils import normalize_values_in_dict
from skmultiflow.classification.core.utils.utils import get_max_value_index
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
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

    This adaptive Hoeffding Tree uses ADWIN to monitor performance of branches on the tree and to replace them with new
    branches when their accuracy decreases if the new branches are more accurate.

        See details in:

        Adaptive Learning from Evolving Data Streams.
        Albert Bifet, Ricard Gavaldà. IDA 2009

    Parameters
    ----------
    TODO


     Examples
    --------
    from skmultiflow.classification.trees.hoeffding_adaptive_tree import HAT
    from skmultiflow.data.file_stream import FileStream
    from skmultiflow.options.file_option import FileOption
    from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    # Setup the File Stream
    opt = FileOption("FILE", "OPT_NAME", "/Users/alessandro/Desktop/scikit-multiflow-master/skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()

    classifier = HAT()
    eval = EvaluatePrequential(pretrain_size=200, max_instances=50000, batch_size=1, n_wait=200, max_time=1000,output_file=None, task_type='classification', show_plot=True, plot_options=['kappa', 'kappa_t', 'performance'])

    eval.eval(stream=stream, classifier=classifier)

    """

    class NewNode(metaclass=ABCMeta):
        """
            Abstract Class to create a New Node for HoeffdingAdaptiveTree(HAT)
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
        def kill_tree_childs(self, hat):
            pass

        @abstractmethod
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
            pass

        @abstractmethod
        def filter_instance_to_leaves(self, X, parent, parent_branch, update_splitter_count, found_nodes=None):
            # CHECK avoid mutable defaults, see:
            # https://docs.quantifiedcode.com/python-anti-patterns/correctness/mutable_default_value_as_argument.html
            pass

    class AdaSplitNode(SplitNode, NewNode):
        def __init__(self, split_test, class_observations, size):
            SplitNode.__init__(self, split_test, class_observations, size)
            self._estimation_error_weight = ADWIN()
            self._alternate_tree = None  # CHECK not HoeffdingTree.Node(), I force alternatetree to be None so that will be that initialized as _new_learning_node (line 154)
            self.error_change = False
            self._random_seed = 1
            self._classifier_random = random.seed(self._random_seed)

        # Override SplitNode
        def calc_byte_size_including_subtree(self):
            byte_size = self.__sizeof__()
            if self._alternate_tree is not None:
                byte_size += self._alternate_tree.calc_byte_size_including_subtree()
            if self._estimation_error_weight is not None:
                byte_size += self._estimation_error_weight.get_length_estimation()

            for child in self._children:
                if child is not None:
                    byte_size += child.calc_byte_size_including_subtree()

            return byte_size

        # Override NewNode
        def number_leaves(self):
            num_of_leaves = 0
            for child in self._children:
                if child is not None:
                    num_of_leaves += child.number_leaves()

            return num_of_leaves

        # Override NewNode
        def get_error_estimation(self):
            return self._estimation_error_weight._estimation

        # Override NewNode
        def get_error_width(self):
            w = 0.0
            if (self.is_null_error() is False):
                w = self._estimation_error_weight._width

            return w

        # Override NewNode
        def is_null_error(self):
            return (self._estimation_error_weight is None)

        # Override NewNode
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):

            true_class = y
            class_prediction = 0

            if (self.filter_instance_to_leaf(X, parent, parent_branch).node) is not None:
                class_prediction = get_max_value_index(
                    self.filter_instance_to_leaf(X, parent, parent_branch).node.get_class_votes(X, hat))

            bl_correct = (true_class == class_prediction)

            if self._estimation_error_weight is None:
                self._estimation_error_weight = ADWIN()

            old_error = self.get_error_estimation()

            # Add element to Adwin
            add = 0.0 if (bl_correct is True) else 1.0

            self._estimation_error_weight.add_element(add)
            # Detect change with Adwin
            self.error_change = self._estimation_error_weight.detected_change()

            if (self.error_change is True and old_error > self.get_error_estimation()):
                self.error_change = False

            #Check condition to build a new alternate tree
            if (self.error_change is True):
                self._alternate_tree = hat._new_learning_node()  # check call to new learning node
                hat._alternateTrees += 1

            #Condition to replace alternate tree
            elif (self._alternate_tree is not None and self._alternate_tree.is_null_error() is False):
                if (self.get_error_width() > error_width_threshold and self._alternate_tree.get_error_width() > error_width_threshold):
                    old_error_rate = self.get_error_estimation()
                    alt_error_rate = self._alternate_tree.get_error_estimation()
                    fDelta = .05
                    fN = 1.0 / self._alternate_tree.get_error_width() + 1.0 / (self.get_error_width())

                    # CHECK
                    bound = 1.0 / math.sqrt(
                        2.0 * old_error_rate * (1.0 - old_error_rate) * math.log(2.0 / fDelta) * fN)
                    # To check, bound never less than (old_error_rate - alt_error_rate)
                    if bound < (old_error_rate - alt_error_rate):
                        hat._active_leaf_node_cnt -= self.number_leaves()
                        hat._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                        self.kill_tree_childs(hat)

                        if parent is not None:
                            parent.set_child(parent_branch, self._alternate_tree)
                        else:
                            hat._tree_root = hat._tree_root.alternateTree
                        hat._switchAlternateTrees += 1
                    elif (bound < alt_error_rate - old_error_rate):
                        if isinstance(self._alternate_tree, HAT.ActiveLearningNode):
                            self._alternate_tree = None
                        elif (isinstance(self._alternate_tree, HAT.ActiveLearningNode)):
                            self._alternate_tree = None
                        else:
                            self._alternate_tree.kill_tree_childs(hat)
                        hat._prunedalternateTree += 1  # hat._pruned_alternate_trees to check

            # Learn_From_Instance alternate Tree and Child nodes
            if self._alternate_tree is not None:
                self._alternate_tree.learn_from_instance(X, y, weight, hat, parent, parent_branch)

            child_branch = self.instance_child_index(X)
            child = self.get_child(child_branch)

            if child is not None:
                child.learn_from_instance(X, y, weight, hat, parent, parent_branch)

        # Override NewNode
        def kill_tree_childs(self, hat):
            for child in self._children:
                if child is not None:
                    # Delete alternate tree if it exists
                    if (isinstance(child, HAT.AdaSplitNode) and child._alternate_tree is not None):
                        self._pruned_alternate_trees += 1
                    # Recursive delete of SplitNodes
                    if isinstance(child, HAT.AdaSplitNode):
                        child.kill_tree_childs(hat)

                    if isinstance(child, HAT.ActiveLearningNode):
                        child = None
                        hat._active_leaf_node_cnt -= 1
                    elif isinstance(child, HAT.InactiveLearningNode):
                        child = None
                        hat._inactive_leaf_node_cnt -= 1

        # override NewNode
        def filter_instance_to_leaves(self, X, parent, parent_branch, update_splitter_counts, found_nodes=None):
            if found_nodes is None:
                found_nodes = []

            child_index = self.instance_child_index(X)

            if child_index >= 0:
                child = self.get_child(child_index)

                if child is not None:
                    child.filter_instance_to_leaves(X, parent, parent_branch, update_splitter_counts, found_nodes)
                else:
                    found_nodes.append(HoeffdingTree.FoundNode(None, self, child_index))
            if self._alternate_tree is not None:
                self._alternate_tree.filter_instance_to_leaves(X, self, -999, update_splitter_counts, found_nodes)

    class AdaLearningNode(LearningNodeNBAdaptive, NewNode):

        def __init__(self, initial_class_observations):
            LearningNodeNBAdaptive.__init__(self, initial_class_observations)
            self.estimationErrorWeight = ADWIN()
            self.ErrorChange = False
            self.randomSeed = 1
            self.classifierRandom = random.seed(self.randomSeed)

        def calc_byte_size(self):
            byte_size = self.__sizeof__()
            if self.estimationErrorWeight is not None:
                byte_size += self.estimationErrorWeight.get_length_estimation()
            return byte_size

        # Override NewNode
        def number_leaves(self):
            return 1

        # Override NewNode
        def get_error_estimation(self):
            return self.estimationErrorWeight._estimation

        # Override NewNode
        def get_error_width(self):
            return self.estimationErrorWeight._width

        # Override NewNode
        def is_null_error(self):
            return (self.estimationErrorWeight is None)

        def kill_tree_childs(self, hat):
            pass

        # Override NewNode
        def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
            true_class = y

            k = np.random.poisson(1.0, self.classifierRandom)
            if k > 0:
                weight = weight * k

            tmp = self.get_class_votes(X, hat)

            class_prediction = get_max_value_index(tmp)

            bl_correct = (true_class == class_prediction)

            if self.estimationErrorWeight is None:
                self.estimationErrorWeight = ADWIN()

            old_error = self.get_error_estimation()

            # Add element to Adwin
            add = 0.0 if (bl_correct is True) else 1.0

            self.estimationErrorWeight.add_element(add)
            # Detect change with Adwin
            self.ErrorChange = self.estimationErrorWeight.detected_change()

            if (self.ErrorChange is True and old_error > self.get_error_estimation()):
                self.ErrorChange = False

            # Update statistics call LearningNodeNBAdaptive
            super().learn_from_instance(X, y, weight, hat)  # CHECK changed self to super

            # call ActiveLearningNode
            weight_seen = self.get_weight_seen()

            if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= hat.grace_period:
                hat._attempt_to_split(self, parent, parent_branch)
                self.set_weight_seen_at_last_split_evaluation(weight_seen)

        # Override LearningNodeNBAdaptive
        def get_class_votes(self, X, ht):

            dist = {}
            prediction_option = ht.leaf_prediction

            # TODO
            # if prediction_option == MAJORITY_CLASS:  # MC
            #   dist = self.get_observed_class_distribution()
            # elif prediction_option == NAIVE_BAYES:  # NB
            #   dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

            # NBAdaptive
            if self._mc_correct_weight > self._nb_correct_weight:
                dist = self.get_observed_class_distribution()
            else:
                dist = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

            dist_sum = sum(dist.values())  # sum all values in dictionary

            if dist_sum * self.get_error_estimation() * self.get_error_estimation() > 0.0:
                normalize_values_in_dict(dist_sum * self.get_error_estimation() * self.get_error_estimation(), dist)

            return dist

        # Override NewNode, New for option votes
        def filter_instance_to_leaves(self, X, split_parent, parent_branch, update_splitter_counts, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
            found_nodes.append(HoeffdingTree.FoundNode(self, split_parent, parent_branch))

    # =============================================
    # == Hoeffding Adaptive Tree implementation ===
    # =============================================

    def __init__(self, *args, **kwargs):

        super(HAT, self).__init__(*args, **kwargs)
        self._alternate_trees = 0
        self._pruned_alternate_trees = 0
        self._switch_alternate_trees = 0
        self._tree_root = None  # CHECK I not utilize _tree_root of HoeffdingTree. OK


    def reset(self):
        self._alternate_trees = 0
        self._pruned_alternate_trees = 0
        self._switch_alternate_trees = 0
        self._tree_root = None

    # Override HoeffdingTree/BaseClassifier
    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit

        Trains the model on samples X and targets y.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            Data instances.

        y: Array-like
            Contains the classification targets for all samples in X.

        classes: Not used.

        weight: Float or Array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        self

        """
        if y is not None:
            if weight is None:
                weight = np.array([1.0])
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(weight)
            if row_cnt != wrow_cnt:
                weight = [weight[0]] * row_cnt
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._partial_fit(X[i], y[i], weight[i])

    # Override HoeffdingTree
    def _partial_fit(self, X, y, weight):

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        self._tree_root.learn_from_instance(X, y, weight, self, None, -1)

    def filter_instance_to_leaves(self, X, split_parent, parent_branch, update_splitter_counts):
        nodes = []
        self._tree_root.filter_instance_to_leaves(X, split_parent, parent_branch, update_splitter_counts, nodes)
        return nodes

    # Override HoeffdingTree
    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_node = self.filter_instance_to_leaves(X, None, -1, False)
            result = {}
            predict_path = 0
            for fn in found_node:
                if fn.parent_branch != -999:
                    leaf_node = fn.node
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)  # add elements to dictionary
            return result
        else:
            return {}

    # Override HoeffdingTree/BaseClassifier
    def predict(self, X):
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i])
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                predictions.append(max(votes, key=votes.get))
        return predictions

    def score(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    # Override HoeffdingTree
    def _new_learning_node(self, initial_class_observations=None):
        return self.AdaLearningNode(initial_class_observations)

    # Override HoeffdingTree
    def new_split_node(self, split_test, class_observations, size):
        return self.AdaSplitNode(split_test, class_observations, size)

