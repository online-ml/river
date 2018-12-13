from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.utils.utils import *
from skmultiflow.trees.utils import do_naive_bayes_prediction


GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'


class LCHT(HoeffdingTree):
    """ Label Combination Hoeffding Tree

    Label combination transforms the problem from multilabel to multiclass.
    For each unique combination of labels it assigns a class and procedes
    with training the hoeffding tree normaly.

    The transformation is done by changing the label set which could be seen
    as a binary number to an int which will represent the class, and after
    the prediction the int is converted back to a bnary number which is the
    predicted labelset.

    The number of labels need to be provided for the transformation to work.

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
    n_labels: int (default=None)
        the number of labels the problem has.
    """
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
                 nominal_attributes=None,
                 n_labels=None):

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
        self.n_labels=n_labels

    @property
    def n_labels(self):
        return self._n_labels

    @n_labels.setter
    def n_labels(self, n_labels):
        if n_labels is None:
            raise ValueError('The number of labels must be specified')
        self._n_labels = n_labels

    def partial_fit(self, X, y, classes=None, weight=None):
        super().partial_fit(X, y, weight=weight)    # Override HT, infer the classes

    class LCActiveLearningNode(HoeffdingTree.ActiveLearningNode):

        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):

            if not(ht.leaf_prediction == NAIVE_BAYES_ADAPTIVE):
                y = ''.join(str(e) for e in y)
                y = int(y, 2)

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

    class LCInactiveLearningNode(HoeffdingTree.InactiveLearningNode):

        def __init__(self, initial_class_observations=None):
            """ LCInactiveLearningNode class constructor. """
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, ht):

            i = ''.join(str(e) for e in y)
            i = int(i, 2)
            try:
                self._observed_class_distribution[i] += weight
            except KeyError:
                self._observed_class_distribution[i] = weight

    class LCLearningNodeNB(LCActiveLearningNode):

        def __init__(self, initial_class_observations):
            """ LCLearningNodeNB class constructor. """
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, ht):
            """Get the votes per class for a given instance.

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
            """Disable an attribute observer.

            Disabled in Nodes using Naive Bayes, since poor attributes are used in Naive Bayes calculation.

            Parameters
            ----------
            att_index: int
                Attribute index.

            """
            pass

    class LCLearningNodeNBA(LCLearningNodeNB):

        def __init__(self, initial_class_observations):
            """LCLearningNodeNBA class constructor. """
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

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

            y = ''.join(str(e) for e in y)
            y = int(y, 2)

            if self._observed_class_distribution == {}:
                # All target_values equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            """Get the votes per class for a given instance.

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

    def predict(self, X):
        """Predicts the label of the X instance(s)

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
            pred = str("{0:0"+str(self.n_labels)+"b}").format(index)
            pred = [int(e) for e in pred]
            predictions.append(pred)

        return np.array(predictions)

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.LCActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LCLearningNodeNB(initial_class_observations)
        else:
            return self.LCLearningNodeNBA(initial_class_observations)

    def _deactivate_learning_node(self, to_deactivate: LCActiveLearningNode, parent: HoeffdingTree.SplitNode, parent_branch: int):
        """Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: ActiveLearningNode
            The node to deactivate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self.LCInactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1