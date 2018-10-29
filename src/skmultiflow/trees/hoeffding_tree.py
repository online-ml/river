import logging
import textwrap
from abc import ABCMeta
from operator import attrgetter
import numpy as np
from skmultiflow.utils.utils import get_dimensions, normalize_values_in_dict, calculate_object_size
from skmultiflow.core.base import StreamModel
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_class_observer_null import AttributeClassObserverNull
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.utils import do_naive_bayes_prediction

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'

# logger
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class HoeffdingTree(StreamModel):
    """ Hoeffding Tree or VFDT.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_criterion: string (default='info_gain')
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
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
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    A Hoeffding Tree [1]_ is an incremental, anytime decision tree induction algorithm that is capable of learning from
    massive data streams, assuming that the distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an optimal splitting attribute. This idea is
    supported mathematically by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision tree learners is that
    it has sound guarantees of performance. Using the Hoeffding bound one can show that its output is asymptotically
    nearly identical to that of a non-incremental learner using infinitely many examples.

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1] G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    """

    class FoundNode(object):
        """ Base class for tree nodes.

        Parameters
        ----------
        node: SplitNode or LearningNode
            The node object.
        parent: SplitNode or None
            The node's parent.
        parent_branch: int
            The parent node's branch.

        """
        def __init__(self, node=None, parent=None, parent_branch=None):
            """ FoundNode class constructor. """
            self.node = node
            self.parent = parent
            self.parent_branch = parent_branch

    class Node(metaclass=ABCMeta):
        """ Base class for nodes in a Hoeffding Tree.

        Parameters
        ----------
        class_observations: dict (class_value, weight) or None
            Class observations.

        """

        def __init__(self, class_observations=None):
            """ Node class constructor. """
            if class_observations is None:
                class_observations = {}  # Dictionary (class_value, weight)
            self._observed_class_distribution = class_observations

        @staticmethod
        def is_leaf():
            """ Determine if the node is a leaf.

            Returns
            -------
            True if leaf, False otherwise

            """
            return True

        def filter_instance_to_leaf(self, X, parent, parent_branch):
            """ Traverse down the tree to locate the corresponding leaf for an instance.

            Parameters
            ----------
            X: numpy.ndarray of shape (n_samples, n_features)
               Data instances.
            parent: HoeffdingTree.Node
                Parent node.
            parent_branch: Int
                Parent branch index

            Returns
            -------
            FoundNode
                The corresponding leaf.

            """
            return HoeffdingTree.FoundNode(self, parent, parent_branch)

        def get_observed_class_distribution(self):
            """ Get the current observed class distribution at the node.

            Returns
            -------
            dict (class_value, weight)
                Class distribution at the node.

            """
            return self._observed_class_distribution

        def get_class_votes(self, X, ht):
            """ Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
               Data instances.
            ht: HoeffdingTree
                The Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            return self._observed_class_distribution

        def observed_class_distribution_is_pure(self):
            """ Check if observed class distribution is pure, i.e. if all samples belong to the same class.

            Returns
            -------
            boolean
                True if observed number of classes is less than 2, False otherwise.

            """
            count = 0
            for _, weight in self._observed_class_distribution.items():
                if weight is not 0:
                    count += 1
                    if count == 2:  # No need to count beyond this point
                        break
            return count < 2

        def subtree_depth(self):
            """ Calculate the depth of the subtree from this node.

            Returns
            -------
            int
                Subtree depth, 0 if the node is a leaf.

            """
            return 0

        def calculate_promise(self):
            """ Calculate node's promise.

            Returns
            -------
            int
                A small value indicates that the node has seen more samples of a given class than the other classes.

            """
            total_seen = sum(self._observed_class_distribution.values())
            if total_seen > 0:
                return total_seen - max(self._observed_class_distribution.values())
            else:
                return 0

        def describe_subtree(self, ht, buffer, indent=0):
            """ Walk the tree and write its structure to a buffer string.

            Parameters
            ----------
            ht: HoeffdingTree
                The tree to describe.
            buffer: string
                The string buffer where the tree's structure will be stored
            indent: int
                Indentation level (number of white spaces for current node.)

            """
            buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)

            try:
                class_val = max(self._observed_class_distribution, key=self._observed_class_distribution.get)
                buffer[0] += 'Class {} | {}\n'.format(class_val, self._observed_class_distribution)
            except ValueError:  # Regression problems
                buffer[0] += 'Statistics {}\n'.format(self._observed_class_distribution)

        # TODO
        def get_description(self):
            pass

    class SplitNode(Node):
        """ Node that splits the data in a Hoeffding Tree.

        Parameters
        ----------
        split_test: InstanceConditionalTest
            Split test.
        class_observations: dict (class_value, weight) or None
            Class observations

        """

        def __init__(self, split_test, class_observations):
            """ SplitNode class constructor."""
            super().__init__(class_observations)
            self._split_test = split_test
            # Dict of tuples (branch, child)
            self._children = {}

        def num_children(self):
            """ Count the number of children for a node."""
            return len(self._children)

        def set_child(self, index, node):
            """ Set node as child.

            Parameters
            ----------
            index: int
                Branch index where the node will be inserted.

            node: HoeffdingTree.Node
                The node to insert.

            """
            if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
                raise IndexError
            self._children[index] = node

        def get_child(self, index):
            """ Retrieve a node's child given its branch index.

            Parameters
            ----------
            index: int
                Node's branch index.

            Returns
            -------
            HoeffdingTree.Node or None
                Child node.

            """
            if index in self._children:
                return self._children[index]
            else:
                return None

        def instance_child_index(self, X):
            """ Get the branch index for a given instance at the current node.

            Returns
            -------
            int
                Branch index, -1 if unknown.

            """
            return self._split_test.branch_for_instance(X)

        @staticmethod
        def is_leaf():
            """ Determine if the node is a leaf.

            Returns
            -------
            boolean
                True if node is a leaf, False otherwise

            """
            return False

        def filter_instance_to_leaf(self, X, parent, parent_branch):
            """ Traverse down the tree to locate the corresponding leaf for an instance.

            Parameters
            ----------
            X: numpy.ndarray of shape (n_samples, n_features)
               Data instances.
            parent: HoeffdingTree.Node
                Parent node.
            parent_branch: int
                Parent branch index.

            Returns
            -------
            FoundNode
                Leaf node for the instance.

            """
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    return child.filter_instance_to_leaf(X, self, child_index)
                else:
                    return HoeffdingTree.FoundNode(None, self, child_index)
            else:
                return HoeffdingTree.FoundNode(self, parent, parent_branch)

        def subtree_depth(self):
            """ Calculate the depth of the subtree from this node.

            Returns
            -------
            int
                Subtree depth, 0 if node is a leaf.
            """
            max_child_depth = 0
            for child in self._children:
                if child is not None:
                    depth = child.subtree_depth()
                    if depth > max_child_depth:
                        max_child_depth = depth
            return max_child_depth + 1

        def describe_subtree(self, ht, buffer, indent=0):
            """ Walk the tree and write its structure to a buffer string.

            Parameters
            ----------
            ht: HoeffdingTree
                The tree to describe.
            buffer: string
                The buffer where the tree's structure will be stored.
            indent: int
                Indentation level (number of white spaces for current node).

            """
            for branch_idx in range(self.num_children()):
                child = self.get_child(branch_idx)
                if child is not None:
                    buffer[0] += textwrap.indent('if ', ' ' * indent)
                    buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                    buffer[0] += ':\n'
                    child.describe_subtree(ht, buffer, indent + 2)

    class LearningNode(Node):
        """ Base class for Learning Nodes in a Hoeffding Tree.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """
        def __init__(self, initial_class_observations=None):
            """ LearningNode class constructor. """
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """
            pass

    class InactiveLearningNode(LearningNode):
        """ Inactive learning node that does not grow.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """
        def __init__(self, initial_class_observations=None):
            """ InactiveLearningNode class constructor. """
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):
            """ Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight

    class ActiveLearningNode(LearningNode):
        """ Learning node that supports growth.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """
        def __init__(self, initial_class_observations):
            """ ActiveLearningNode class constructor. """
            super().__init__(initial_class_observations)
            self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
            self._attribute_observers = {}

        def learn_from_instance(self, X, y, weight, ht):
            """ Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = NumericAttributeClassObserverGaussian()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            """ Calculate the total weight seen by the node.

            Returns
            -------
            float
                Total weight seen.

            """
            return sum(self._observed_class_distribution.values())

        def get_weight_seen_at_last_split_evaluation(self):
            """ Retrieve the weight seen at last split evaluation.

            Returns
            -------
            float
                Weight seen at last split evaluation.

            """
            return self._weight_seen_at_last_split_evaluation

        def set_weight_seen_at_last_split_evaluation(self, weight):
            """ Retrieve the weight seen at last split evaluation.

            Parameters
            ----------
            weight: float
                Weight seen at last split evaluation.

            """
            self._weight_seen_at_last_split_evaluation = weight

        def get_best_split_suggestions(self, criterion, ht):
            """ Find possible split candidates.

            Parameters
            ----------
            criterion: SplitCriterion
                The splitting criterion to be used.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            list
                Split candidates.

            """
            best_suggestions = []
            pre_split_dist = self._observed_class_distribution
            if not ht.no_preprune:
                # Add null split as an option
                null_split = AttributeSplitSuggestion(None, [{}],
                                                      criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
                best_suggestions.append(null_split)
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, ht.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
            return best_suggestions

        def disable_attribute(self, att_idx):
            """ Disable an attribute observer.

            Parameters
            ----------
            att_idx: int
                Attribute index.

            """
            if att_idx in self._attribute_observers:
                self._attribute_observers[att_idx] = AttributeClassObserverNull()

    class LearningNodeNB(ActiveLearningNode):
        """ Learning node that uses Naive Bayes models.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """
        def __init__(self, initial_class_observations):
            """ LearningNodeNB class constructor. """
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, ht):
            """ Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self.get_weight_seen() >= ht.nb_threshold:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, ht)

        def disable_attribute(self, att_index):
            """ Disable an attribute observer.

            Disabled in Nodes using Naive Bayes, since poor attributes are used in Naive Bayes calculation.

            Parameters
            ----------
            att_index: int
                Attribute index.

            """
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        """ Learning node that uses Adaptive Naive Bayes models.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations

        """
        def __init__(self, initial_class_observations):
            """ LearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """ Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                The instance's weight.
            ht: HoeffdingTree
                The Hoeffding Tree to update.

            """
            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            """ Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    # ====================================
    # == Hoeffding Tree implementation ===
    # ====================================
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
        """ HoeffdingTree class constructor."""
        super().__init__()
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        # self._numeric_estimator_option = 'GaussianNumericAttributeClassObserver'           # Numeric estimator to use.
        # self._nominal_estimator_option = 'NominalAttributeClassObserver'                   # Nominal estimator to use.

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0
        self.classes = None

    @property
    def max_byte_size(self):
        return self._max_byte_size

    @max_byte_size.setter
    def max_byte_size(self, max_byte_size):
        self._max_byte_size = max_byte_size

    @property
    def memory_estimate_period(self):
        return self._memory_estimate_period

    @memory_estimate_period.setter
    def memory_estimate_period(self, memory_estimate_period):
        self._memory_estimate_period = memory_estimate_period

    @property
    def grace_period(self):
        return self._grace_period

    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion != GINI_SPLIT and split_criterion != INFO_GAIN_SPLIT:
            logger.info("Invalid option {}', will use default '{}'".format(split_criterion, INFO_GAIN_SPLIT))
            self._split_criterion = INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @property
    def split_confidence(self):
        return self._split_confidence

    @split_confidence.setter
    def split_confidence(self, split_confidence):
        self._split_confidence = split_confidence

    @property
    def tie_threshold(self):
        return self._tie_threshold

    @tie_threshold.setter
    def tie_threshold(self, tie_threshold):
        self._tie_threshold = tie_threshold

    @property
    def binary_split(self):
        return self._binary_split

    @binary_split.setter
    def binary_split(self, binary_split):
        self._binary_split = binary_split

    @property
    def stop_mem_management(self):
        return self._stop_mem_management

    @stop_mem_management.setter
    def stop_mem_management(self, stop_mem_management):
        self._stop_mem_management = stop_mem_management

    @property
    def remove_poor_atts(self):
        return self._remove_poor_atts

    @remove_poor_atts.setter
    def remove_poor_atts(self, remove_poor_atts):
        self._remove_poor_atts = remove_poor_atts

    @property
    def no_preprune(self):
        return self._no_preprune

    @no_preprune.setter
    def no_preprune(self, no_pre_prune):
        self._no_preprune = no_pre_prune

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction != MAJORITY_CLASS and leaf_prediction != NAIVE_BAYES \
                and leaf_prediction != NAIVE_BAYES_ADAPTIVE:
            logger.info("Invalid option {}', will use default '{}'".format(leaf_prediction, NAIVE_BAYES_ADAPTIVE))
            self._leaf_prediction = NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def nb_threshold(self):
        return self._nb_threshold

    @nb_threshold.setter
    def nb_threshold(self, nb_threshold):
        self._nb_threshold = nb_threshold

    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, nominal_attributes):
        if nominal_attributes is None:
            nominal_attributes = []
            logger.debug("No Nominal attributes have been defined, will consider all attributes as numerical.")
        self._nominal_attributes = nominal_attributes

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    def measure_byte_size(self):
        """ Calculate the size of the tree.

        Returns
        -------
        int
            Size of the tree in bytes.

        """
        return calculate_object_size(self)

    def reset(self):
        """ Reset the Hoeffding Tree to default values."""
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        if self._leaf_prediction != MAJORITY_CLASS:
            self._remove_poor_atts = None
        self._train_weight_seen_by_model = 0.0

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes and their
        corresponding targets y.

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for the instance and update the leaf node
          statistics.
        * If growth is allowed and the number of instances that the leaf has observed between split attempts
          exceed the grace period then attempt to split.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            if weight is None:
                weight = np.array([1.0])
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(weight)
            if row_cnt != wrow_cnt:
                weight = [weight[0]] * row_cnt
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._train_weight_seen_by_model += weight[i]
                    self._partial_fit(X[i], y[i], weight[i])

    def _partial_fit(self, X, y, weight):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
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
                if weight_diff >= self.grace_period:
                    self._attempt_to_split(active_learning_node, found_node.parent, found_node.parent_branch)
                    active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()

    def get_votes_for_instance(self, X):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.get_class_votes(X, self)
        else:
            return {}

    def predict(self, X):
        """ Predicts the label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)

    def predict_proba(self, X):
        """ Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i]).copy()
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(votes.values()) != 0:
                    normalize_values_in_dict(votes)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        return np.array(predictions)

    @property
    def get_model_measurements(self):
        """ Collect metrics corresponding to the current status of the tree.

        Returns
        -------
        string
            A string buffer containing the measurements of the tree.
        """
        measurements = {'Tree size (nodes)': self._decision_node_cnt
                                               + self._active_leaf_node_cnt
                                               + self._inactive_leaf_node_cnt,
                        'Tree size (leaves)': self._active_leaf_node_cnt + self._inactive_leaf_node_cnt,
                        'Active learning nodes': self._active_leaf_node_cnt, 'Tree depth': self.measure_tree_depth(),
                        'Active leaf byte size estimate': self._active_leaf_byte_size_estimate,
                        'Inactive leaf byte size estimate': self._inactive_leaf_byte_size_estimate,
                        'Byte size estimate overhead': self._byte_size_estimate_overhead_fraction
                        }
        return measurements

    def measure_tree_depth(self):
        """ Calculate the depth of the tree.

        Returns
        -------
        int
            Depth of the tree.
        """
        if isinstance(self._tree_root, self.Node):
            return self._tree_root.subtree_depth()
        return 0

    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.ActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations)

    def get_model_description(self):
        """ Walk the tree and return its structure in a buffer.

        Returns
        -------
        string
            The description of the model.

        """
        if self._tree_root is not None:
            buffer = ['']
            description = ''
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description

    @staticmethod
    def compute_hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each node.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an information gain the range
        is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct attribute at any given node.

        :math:`n`: Number of samples.

        Parameters
        ----------
        range_val: float
            Range value.
        confidence: float
            Confidence of choosing the correct attribute.
        n: int
            Number of samples.

        Returns
        -------
        float
            The Hoeffding bound.

        """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def new_split_node(self, split_test, class_observations):
        """ Create a new split node."""
        return self.SplitNode(split_test, class_observations)

    def _attempt_to_split(self, node: ActiveLearningNode, parent: SplitNode, parent_idx: int):
        """ Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node: ActiveLearningNode
            The node to evaluate.
        parent: SplitNode
            The node's parent.
        parent_idx: int
            Parent node's branch index.

        """
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):    # best_suggestion.merit > 1e-10 and \
                    should_split = True
                if self.remove_poor_atts is not None and self.remove_poor_atts:
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
                if split_decision.split_test is None:
                    # Preprune - null wins
                    self._deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self.new_split_node(split_decision.split_test,
                                                    node.get_observed_class_distribution())
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
        """ Track the size of the tree and disable/enable nodes if required."""
        byte_size = (self._active_leaf_byte_size_estimate
                     + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate) \
                    * self._byte_size_estimate_overhead_fraction
        if self._inactive_leaf_node_cnt > 0 or byte_size > self.max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        learning_nodes = self._find_learning_nodes()
        learning_nodes.sort(key=lambda n: n.node.calculate_promise())
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if ((max_active * self._active_leaf_byte_size_estimate + (len(learning_nodes) - max_active)
                * self._inactive_leaf_byte_size_estimate) * self._byte_size_estimate_overhead_fraction) \
                    > self.max_byte_size:
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if isinstance(learning_nodes[i].node, self.ActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)
        for i in range(cutoff, len(learning_nodes)):
            if isinstance(learning_nodes[i].node, self.InactiveLearningNode):
                self._activate_learning_node(learning_nodes[i].node,
                                             learning_nodes[i].parent,
                                             learning_nodes[i].parent_branch)

    def estimate_model_byte_size(self):
        """ Calculate the size of the model and trigger tracker function if the actual model size exceeds the max size
        in the configuration."""
        learning_nodes = self._find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if isinstance(found_node.node, self.ActiveLearningNode):
                total_active_size += calculate_object_size(found_node.node)
            else:
                total_inactive_size += calculate_object_size(found_node.node)
        if total_active_size > 0:
            self._active_leaf_byte_size_estimate = total_active_size / self._active_leaf_node_cnt
        if total_inactive_size > 0:
            self._inactive_leaf_byte_size_estimate = total_inactive_size / self._inactive_leaf_node_cnt
        actual_model_size = calculate_object_size(self)
        estimated_model_size = (self._active_leaf_node_cnt * self._active_leaf_byte_size_estimate
                                + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate)
        self._byte_size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self.max_byte_size:
            self.enforce_tracker_limit()

    def deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for i in range(len(learning_nodes)):
            if isinstance(learning_nodes[i], self.ActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)

    def _deactivate_learning_node(self, to_deactivate: ActiveLearningNode, parent: SplitNode, parent_branch: int):
        """ Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: ActiveLearningNode
            The node to deactivate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self.InactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def _activate_learning_node(self, to_activate: InactiveLearningNode, parent: SplitNode, parent_branch: int):
        """ Activate a learning node.

        Parameters
        ----------
        to_activate: InactiveLearningNode
            The node to activate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(to_activate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def _find_learning_nodes(self):
        """ Find learning nodes in the tree.

        Returns
        -------
        list
            List of learning nodes in the tree.
        """
        found_list = []
        self.__find_learning_nodes(self._tree_root, None, -1, found_list)
        return found_list

    def __find_learning_nodes(self, node, parent, parent_branch, found):
        """ Find learning nodes in the tree from a given node.

        Parameters
        ----------
        node: HoeffdingTree.Node
            The node to start the search.
        parent: LearningNode or SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch.

        Returns
        -------
        list
            List of learning nodes.
        """
        if node is not None:
            if isinstance(node, self.LearningNode):
                found.append(self.FoundNode(node, parent, parent_branch))
            if isinstance(node, self.SplitNode):
                split_node = node
                for i in range(split_node.num_children()):
                    self.__find_learning_nodes(split_node.get_child(i), split_node, i, found)

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        """ Collect information about the Hoeffding Tree configuration.

        Returns
        -------
        string
            Configuration for the Hoeffding Tree.
        """
        description = type(self).__name__ + ': '
        description += 'max_byte_size: {} - '.format(self.max_byte_size)
        description += 'memory_estimate_period: {} - '.format(self.memory_estimate_period)
        description += 'grace_period: {} - '.format(self.grace_period)
        description += 'split_criterion: {} - '.format(self.split_criterion)
        description += 'split_confidence: {} - '.format(self.split_confidence)
        description += 'tie_threshold: {} - '.format(self.tie_threshold)
        description += 'binary_split: {} - '.format(self.binary_split)
        description += 'stop_mem_management: {} - '.format(self.stop_mem_management)
        description += 'remove_poor_atts: {} - '.format(self.remove_poor_atts)
        description += 'no_pre_prune: {} - '.format(self.no_preprune)
        description += 'leaf_prediction: {} - '.format(self.leaf_prediction)
        description += 'nb_threshold: {} - '.format(self.nb_threshold)
        description += 'nominal_attributes: {} - '.format(self.nominal_attributes)
        return description
