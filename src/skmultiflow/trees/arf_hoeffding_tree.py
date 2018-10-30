import numpy as np
from skmultiflow.trees.hoeffding_tree import HoeffdingTree, MAJORITY_CLASS, NAIVE_BAYES
from skmultiflow.utils import check_random_state
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.utils.utils import get_dimensions
from skmultiflow.trees.utils import do_naive_bayes_prediction


class ARFHoeffdingTree(HoeffdingTree):
    """ Adaptive Random Forest Hoeffding Tree.

    Parameters
    __________
    max_byte_size: int, optional (default=33554432)
        Maximum memory consumed by the tree.

    memory_estimate_period: int, optional (default=2000000)
        Number of instances between memory consumption checks.

    grace_period: int, optional (default=50)
        Number of instances a leaf should observe between split attempts.

    split_criterion: string, optional (default='info_gain')
        Split criterion to use.

        - 'gini' - Gini
        - 'info_gain' - Information Gain

    split_confidence: float, optional (default=0.01)
        Allowed error in split decision, a value closer to 0 takes longer to decide.

    tie_threshold: float, optional (default=0.05)
        Threshold below which a split will be forced to break ties.

    binary_split: bool, optional (default=False)
        If True, only allow binary splits.

    stop_mem_management: bool, optional (default=False)
        If True, stop growing as soon as memory limit is hit.

    remove_poor_atts: bool, optional (default=False)
        If True, disable poor attributes.

    no_preprune: bool, optional (default=False)
        If True, disable pre-pruning.

    leaf_prediction: string, optional (default='nba')
        Prediction mechanism used at leafs.

        - 'mc' - Majority Class
        - 'nb' - Naive Bayes
        - 'nba' - Naive Bayes Adaptive

    nb_threshold: int, optional (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.

    Notes
    _____
    This is the base model for the Adaptive Random Forest ensemble learner
    (See skmultiflow.classification.meta.adaptive_random_forests).
    This Hoeffding Tree includes a max_features parameter, which defines the number of randomly selected features to
    be considered at each split.

    """
    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):
        """Random learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations.

        max_features: int
            Number of attributes per subset for each node split.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.

        """
        def __init__(self, initial_class_observations, max_features, random_state=None):
            """ RandomLearningNode class constructor. """
            super().__init__(initial_class_observations)

            self.max_features = max_features
            self._attribute_observers = {}
            self.list_attributes = np.array([])
            self.random_state = random_state

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
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
            if self.list_attributes.size == 0:
                self.list_attributes = self._sample_features(get_dimensions(X)[1])

            for i in self.list_attributes:
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = NumericAttributeClassObserverGaussian()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def _sample_features(self, n_features):
            return self.random_state.choice(n_features, size=self.max_features, replace=False)

    class LearningNodeNB(RandomLearningNode):
        """Naive Bayes learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations.

        max_features: int
            Number of attributes per subset for each node split.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.

        """
        def __init__(self, initial_class_observations, max_features, random_state):
            """ LearningNodeNB class constructor. """
            super().__init__(initial_class_observations, max_features, random_state)

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

    class LearningNodeNBAdaptive(LearningNodeNB):
        """Naive Bayes Adaptive learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations.

        max_features: int
            Number of attributes per subset for each node split.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.

        """
        def __init__(self, initial_class_observations, max_features, random_state):
            """LearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations, max_features, random_state)
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

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.01,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 max_features=2,
                 random_state=None):
        """ARFHoeffdingTree class constructor."""
        super().__init__(max_byte_size,
                         memory_estimate_period,
                         grace_period,
                         split_criterion,
                         split_confidence,
                         tie_threshold,
                         binary_split,
                         stop_mem_management,
                         remove_poor_atts,
                         no_preprune,
                         leaf_prediction,
                         nb_threshold,
                         nominal_attributes)
        self.max_features = max_features
        self.remove_poor_attributes = False
        self.random_state = check_random_state(random_state)

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        # MAJORITY CLASS
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.max_features,
                                           random_state=self.random_state)
        # NAIVE BAYES
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.max_features,
                                       random_state=self.random_state)
        # NAIVE BAYES ADAPTIVE
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations, self.max_features,
                                               random_state=self.random_state)

    @staticmethod
    def is_randomizable():
        return True

    def new_instance(self):
        return ARFHoeffdingTree(max_byte_size=self.max_byte_size,
                                memory_estimate_period=self.memory_estimate_period,
                                grace_period=self.grace_period,
                                split_criterion=self.split_criterion,
                                split_confidence=self.split_confidence,
                                tie_threshold=self.tie_threshold,
                                binary_split=self.binary_split,
                                stop_mem_management=self.stop_mem_management,
                                remove_poor_atts=self.remove_poor_atts,
                                no_preprune=self.no_preprune,
                                leaf_prediction=self.leaf_prediction,
                                nb_threshold=self.nb_threshold,
                                nominal_attributes=self.nominal_attributes,
                                max_features=self.max_features,
                                random_state=self.random_state)
