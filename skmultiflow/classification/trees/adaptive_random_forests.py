__author__ = 'Anderson Carlos Ferreira da Silva'

from random import randint
from skmultiflow.core.base_object import BaseObject
from skmultiflow.classification.trees.hoeffding_tree import *
from skmultiflow.classification.core.driftdetection.adwin import ADWIN

INSTANCE_WEIGHT = np.array([1.0])
FEATURE_MODE_M = ''
FEATURE_MODE_SQRT = 'sqrt'
FEATURE_MODE_SQRT_INV = 'sqrt_inv'
FEATURE_MODE_PERCENTAGE = 'percentage'


# ## ADFHoeffdingTree

# ### References

# - [Hoeffding Tree](https://github.com/scikit-multiflow/scikit-multiflow/blob/17327dc81b7d6e35d533795ae13493ad08118708/skmultiflow/classification/trees/hoeffding_tree.py)
# - [Adaptive Random Forest Hoeffding Tree](https://github.com/Waikato/moa/blob/f5cdc1051a7247bb61702131aec3e62b40aa82f8/moa/src/main/java/moa/classifiers/trees/ARFHoeffdingTree.java)

class ARFHoeffdingTree(HoeffdingTree):
            
    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):                    
        """Random learning node.
        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        """
        def __init__(self, initial_class_observations, nb_attributes):
            super().__init__(initial_class_observations)
            self.nb_attributes = nb_attributes
            self._attribute_observers = [None] * nb_attributes
            self.list_attributes = []         
            
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
            if y not in self._observed_class_distribution:
                self._observed_class_distribution[y] = 0.0            
            
            self._observed_class_distribution[y] += weight                            
            if not self.list_attributes:
                self.list_attributes = [None] * self.nb_attributes
                for j in range(self.nb_attributes):    
                    is_unique = False
                    while is_unique == False:
                        self.list_attributes[j] = randint(0, self.nb_attributes - 1)
                        is_unique = True
                        for i in range(j):
                            if self.list_attributes[j] == self.list_attributes[i]:
                                is_unique = False
                                break

            for j in range(self.nb_attributes):
                i = self.list_attributes[j]
                obs = self._attribute_observers[i]
                if obs is None:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)
            
    class LearningNodeNB(RandomLearningNode):

        def __init__(self, initial_class_observations, nb_attributes):
            super().__init__(initial_class_observations, nb_attributes)            
            
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
        """Learning node that uses Adaptive Naive Bayes models.
        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        """
        def __init__(self, initial_class_observations, nb_attributes):
            """LearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations, nb_attributes)
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
        
    def __init__(self, max_byte_size = 33554432, memory_estimate_period = 1000000, grace_period = 200,
                     split_criterion = 'info_gain', split_confidence = 0.0000001, tie_threshold = 0.05, 
                     binary_split = False, stop_mem_management = False, remove_poor_atts = False, no_preprune = False, 
                     leaf_prediction = 'nba', nb_threshold = 0, nominal_attributes = None, nb_attributes = 2):
        """ADFHoeffdingTree class constructor."""
        super().__init__(max_byte_size, memory_estimate_period, grace_period, split_criterion, split_confidence,
                        tie_threshold, binary_split, stop_mem_management, remove_poor_atts, no_preprune,
                        leaf_prediction, nb_threshold, nominal_attributes)
        self.nb_attributes = nb_attributes
        self.remove_poor_attributes_option = None        

    def _new_learning_node(self, initial_class_observations = None):   
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}        
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.nb_attributes)            
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.nb_attributes)            
        else: #NAIVE_BAYES_ADAPTIVE
            return self.LearningNodeNBAdaptive(initial_class_observations, self.nb_attributes)
        
    def is_randomizable():  
        return True
    
    def copy(self):
        return ARFHoeffdingTree(nominal_attributes=self.nominal_attributes, nb_attributes = self.nb_attributes)


# ## Adaptive Random Forest

# - [Adaptive Random Forest](https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/meta/AdaptiveRandomForest.java)

class AdaptiveRandomForest(BaseClassifier):
    
    def __init__(self, nb_ensemble = 10, feature_mode = 'sqrt', nb_attributes = 2,
                 disable_background_learner = False, disable_drift_detection = False, 
                 disable_weighted_vote = False, w = 6, evaluator_method = None,
                 drift_detection_method = ADWIN, warning_detection_method = ADWIN,
                 drift_delta = 0.001, warning_delta = 0.01, nominal_atributes=None):
        
        """AdaptiveRandomForest class constructor."""
        super().__init__()          
        self.nb_ensemble = nb_ensemble        
        self.feature_mode = feature_mode
        self.total_attributes = nb_attributes
        self.disable_background_learner = disable_background_learner   
        self.disable_drift_detection = disable_drift_detection        
        self.disable_weighted_vote = disable_weighted_vote
        self.w = w
        self.evaluator_method = ARFBaseClassifierEvaluator
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self.nb_attributes = None
        self.ensemble = None
        self.warning_delta = warning_delta
        self.drift_delta = drift_delta
        self.nominal_attributes = nominal_atributes

    def fit(self, X, y, classes = None, weight = None):
        raise NotImplementedError
    
    def partial_fit(self, X, y, classes = None, weight = None):
        if y is not None:
            if weight is None:
                weight = INSTANCE_WEIGHT
            row_cnt, _ = get_dimensions(X)
            wrow_cnt, _ = get_dimensions(weight)
            if row_cnt != wrow_cnt:
                weight = [weight[0]] * row_cnt                
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._train_weight_seen_by_model += weight[i]
                    self._partial_fit(X[i], y[i], weight[i])
        
    def _partial_fit(self, X, y, weight):
        self.instances_seen += 1
        
        if self.ensemble is None:
            self.init_ensemble(X)
                      
        for i in range(self.nb_ensemble):
            y_predicted = self.ensemble[i].predict(np.asarray([X]))
            self.ensemble[i].evaluator.update(y_predicted, np.asarray([y]), weight)
            k = np.random.poisson(self.w)
            if k > 0:
                self.ensemble[i].partial_fit(np.asarray([X]), np.asarray([y]), np.asarray([k]), self.instances_seen)
    
    def predict(self, X):
        """Predicts the label of the X instance(s)
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.
        Returns
        -------
        list
            Predicted labels for all instances in X.
        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i])
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                predict = []                
                for vote in votes:                                        
                    predict.append(max(vote, key = vote.get))            
                y, counts = np.unique(predict, return_counts = True)
                value = np.argmax(counts)                
                predictions.append(y[value])                
        return predictions

    def predict_proba(self, X):
        raise NotImplementedError
        
    def reset(self):        
        """Reset attributes."""
        self.ensemble = None
        self.nb_attributes = 0
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self.evaluator_method = ARFBaseClassifierEvaluator
        
    def score(self, X, y):
        raise NotImplementedError
        
    def get_info(self):
        return "NotImplementedError"
        
    def get_votes_for_instance(self, X):
        if self.ensemble is None:
            self.init_ensemble(X)
        combined_votes = []
           
        for i in range(self.nb_ensemble):
            vote = self.ensemble[i].get_votes_for_instance(X)
            if sum(vote) > 0:
                combined_votes.append(vote)
        
        return combined_votes
        
    def init_ensemble(self, X):
        self.ensemble = [None] * self.nb_ensemble
        
        self.nb_attributes = self.total_attributes
        
        """The m (total number of attributes) depends on:"""
        _, n = get_dimensions(X)
        
        if self.feature_mode == FEATURE_MODE_SQRT:
            self.nb_attributes = int(round(math.sqrt(n)) + 1)            
        elif self.feature_mode == FEATURE_MODE_SQRT_INV:
            self.nb_attributes = n - int(round(math.sqrt(n) + 1))
        elif self.feature_mode == FEATURE_MODE_PERCENTAGE:            
            percent = (100 + self.nb_attributes) / 100.0 if self.nb_attributes < 0 else self.nb_attributes / 100.0
            self.nb_attributes = int(round(n * percent))
            
        """Notice that if the selected feature_mode was FEATURE_MODE_M then nothing is performed, 
        still it is necessary to check (and adjusted) for when a negative value was used. 
        """
        
        """m is negative, use size(features) + -m"""
        if self.nb_attributes < 0:
            self.nb_attributes += n
        """Other sanity checks to avoid runtime errors."""
        """m <= 0 (m can be negative if nb_attributes was negative and abs(m) > n), then use m = 1"""
        if self.nb_attributes <= 0:
            self.nb_attributes = 1
        """m > n, then it should use n"""
        if self.nb_attributes > n:
            self.nb_attributes = n
                               
        for i in range(self.nb_ensemble):            
            self.ensemble[i] = ARFBaseLearner(i,
                                              ARFHoeffdingTree(nominal_attributes=self.nominal_attributes,
                                                               nb_attributes = self.nb_attributes),
                                              self.instances_seen,
                                              not self.disable_background_learner,
                                              not self.disable_drift_detection,
                                              self.evaluator_method,
                                              self.drift_detection_method,
                                              self.warning_detection_method,
                                              self.drift_delta,
                                              self.warning_delta,
                                              False)            
                    
    def is_randomizable():  
        return True                  


class ARFBaseLearner(BaseObject):
    
    def __init__(self, index_original, classifier, instances_seen, use_background_learner, use_drift_detector,
                 evaluator_method, drift_detection_method, warning_detection_method,
                 drift_delta, warning_delta, is_background_learner):
        self.index_original = index_original
        self.classifier = classifier 
        self.created_on = instances_seen
        self.use_background_learner = use_background_learner
        self.use_drift_detector = use_drift_detector
        self.is_background_learner = is_background_learner
        self.evaluator_method = evaluator_method
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method
        self.warning_delta = warning_delta
        self.drift_delta = drift_delta
                                   
        self.last_drift_on = 0
        self.last_warning_on = 0
        self.nb_drifts_detected = 0
        self.nb_warnings_detected = 0            

        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        
        self.evaluator = evaluator_method()

        if use_background_learner:
            self.warning_detection = warning_detection_method(self.warning_delta)
            
        if use_drift_detector:
            self.drift_detection = drift_detection_method(self.drift_delta)
            
    def reset(self, instances_seen):
        if self.use_background_learner and self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.evaluator_method = self.background_learner.evaluator_method
            self.created_on = self.background_learner.created_on                
            self.background_learner = None
        else:
            self.classifier.reset()
            self.created_on = instances_seen
            self.drift_detection = self.drift_detection_method(self.drift_delta)
        self.evaluator = self.evaluator_method()

    def partial_fit(self, X, y, weight, instances_seen):
        self.classifier.partial_fit(X, y, weight)

        if self.background_learner:
            self.background_learner.classifier.partial_fit(X, y, INSTANCE_WEIGHT)

        if self.use_drift_detector and not self.is_background_learner:
            correctly_classifies = self.classifier.predict(X) == y
            # Check for warning only if use_background_learner is active
            if self.use_background_learner:
                self.warning_detection.add_element(int(not correctly_classifies))
                # Check if there was a change
                if self.warning_detection.detected_change():
                    self.last_warning_on = instances_seen
                    self.nb_warnings_detected += 1
                    # Create a new background tree classifier
                    background_learner = self.classifier.copy()
                    background_learner.reset() 
                    # Create a new background learner object
                    self.background_learner = ARFBaseLearner(self.index_original, background_learner, instances_seen,
                                                             self.use_background_learner, self.use_drift_detector,
                                                             self.evaluator_method, self.drift_detection_method,
                                                             self.warning_detection_method,
                                                             self.drift_delta, self.warning_delta,
                                                             True)
                    """Update the warning detection object for the current object 
                    (this effectively resets changes made to the object while it was still a bkg learner). 
                    """
                    self.warning_detection = self.warning_detection_method(self.warning_delta)

        # Update the drift detection
        self.drift_detection.add_element(int(not correctly_classifies))

        # Check if there was a change
        if self.drift_detection.detected_change():
            self.last_drift_on = instances_seen
            self.nb_drifts_detected += 1
            self.reset(instances_seen)

    def predict(self, X):
        return self.classifier.predict(X)
    
    def get_votes_for_instance(self, X):
        return self.classifier.get_votes_for_instance(X)

    def get_class_type(self):
        raise NotImplementedError

    def get_info(self):
        return "NotImplementedError"


# - [Basic Classification Performance Evaluator](https://github.com/Waikato/moa/blob/8405512f6131260b3aab042d407a0afd41447b45/moa/src/main/java/moa/evaluation/BasicClassificationPerformanceEvaluator.java)


class ARFBaseClassifierEvaluator(BaseObject):
    
    def __init__(self):
        self.aggregation = 0
        self.length = 0
        
    def update(self, y_predicted, y, weight):
        if weight > 0: 
            self.aggregation += weight if y_predicted == y else 0

    def get_performance(self):
        return self.aggregation * 100
    
    def reset(self):
        self.aggregation = 0
        self.length = 0
    
    def get_class_type(self):
        raise NotImplementedError

    def get_info(self):
        return NotImplementedError
