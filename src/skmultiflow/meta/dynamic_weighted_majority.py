import copy as cp
import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.bayes import NaiveBayes

class DynamicWeightedMajority(StreamModel):
    """
    Dynamic Weighted Majority Ensemble [1]_.

    Parameters
    ----------
    n_estimators: int (default=5)
        Maximum number of estimators to hold.
    base_estimator: StreamModel or sklearn.BaseEstimator (default=NaiveBayes)
        Each member of the ensemble is an instance of the base estimator.
    period: int (default=50)
        Period between expert removal, creation, and weight update.
    beta: float (default=0.5)
        Factor for which to decrease weights by.
    theta: float (default=0.01)
        Minimum fraction of weight per model.

    Notes
    -----
    The method, dynamic weighted majority (DWM), uses four mechanisms to
    cope with concept drift: It trains online learners of the ensemble,
    it weights those learners based on their performance, it removes them,
    also based on their performance, and it adds new experts based on the
    global performance of the ensemble.

    References
    ----------
    .. [1] Kolter and Maloof. Dynamic weighted majority: An ensemble method for drifting concepts.
        The Journal of Machine Learning Research, 8:2755-2790, December 2007. ISSN 1532-4435.
    """

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its weight.

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: float
            The estimator's weight.
        """
        def __init__(self, estimator, weight):
            self.estimator = estimator
            self.weight = weight

        def __lt__(self, other):
            self.weight < other.weight

    def __init__(self, n_estimators=5, base_estimator=NaiveBayes(), period=50, beta=0.5, theta=0.01):
        """
        Creates a new instance of DynamicWeightedMajority.
        """
        super().__init__()

        self.max_experts = n_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.theta = theta
        self.period = period

        self.reset()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError        

    def partial_fit(self, X, y, classes=None, weight=None):
        """
        Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one 
        sample are passed, the algorithm will partial fit the model one sample 
        at a time.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features) 
            Features matrix used for partially updating the model.
            
        y: Array-like
            An array-like of all the class labels for the samples in X.
            
        classes: list 
            List of all existing classes. This is an optional parameter, except
            for the first partial_fit call, when it becomes obligatory.

        weight: None
            Instance weight. This is ignored by the ensemble and is only
            for compliance with the general skmultiflow interface.

        Returns
        -------
        DynamicWeightedMajority
            self
        """
        for i in range(len(X)):
            self.fit_single_sample(
                X[i:i+1, :], y[i:i+1], classes, weight
            )
        return self

    def predict(self, X):
        """ predict
        
        The predict function will take an average of the precitions of its learners,
        weighted by their respective weights, and return the most likely class.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        
        """
        return np.array([np.argmax(self.predict_proba(X))])

    def predict_proba(self, X):
        """ predict_proba
        
        Predicts the probability of each sample belonging to each one of the 
        known classes.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index.
        """
        return self._aggregate_expert_predictions(self.get_expert_predictions(X))

    def fit_single_sample(self, X, y, classes=None, weight=None):
        """
        Fits a single sample of shape X.shape=(1, n_attributes) and y.shape=(1,)

        Aggregates all experts' predictions, diminishes weight of experts whose
        predictions were wrong, and may create or remove experts every n samples.

        Finally, trains each individual expert on the provided data.

        Train loop as described by Kolter and Maloof [1]_.


        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features) 
            Features matrix used for partially updating the model.
            
        y: Array-like
            An array-like of all the class labels for the samples in X.
            
        classes: list 
            List of all existing classes. This is an optional parameter.

        weight: None
            Instance weight. This is ignored by the ensemble and is only
            for compliance with the general skmultiflow interface.
        """
        self.epochs += 1
        self.num_classes = max(len(classes) if classes is not None else 0, (int(np.max(y)) + 1), self.num_classes)
        predictions = np.zeros((self.num_classes,))
        max_weight = 0
        weakest_expert_weight = 1
        weakest_expert_index = None
        
        for i, exp in enumerate(self.experts):
            yHat = exp.estimator.predict(X)
            if np.any(yHat != y) and (self.epochs % self.period == 0):
                exp.weight *= self.beta
            
            predictions[yHat] += exp.weight
            max_weight = max(max_weight, exp.weight)

            if exp.weight < weakest_expert_weight:
                weakest_expert_index = i
                weakest_expert_weight = exp.weight

        yHat = np.array([np.argmax(predictions)])
        if self.epochs % self.period == 0:
            self._scale_weights(max_weight)
            self._remove_experts()
            if np.any(yHat != y):
                if len(self.experts) == self.max_experts:
                    self.experts.pop(weakest_expert_index)
                self.experts.append(self._construct_new_expert())

        ## Train individual experts
        for exp in self.experts:
            exp.estimator.partial_fit(X, y, classes, weight)

    def get_expert_predictions(self, X, classes=None, weight=None):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts,)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2 ## Minimum of 2 classes
        self.experts = [
            self._construct_new_expert()
        ]

    def _scale_weights(self, maxWeight):
        """
        Scales the experts' weights such that the max is 1.
        """
        scaleFactor = 1 / maxWeight
        for exp in self.experts:
            exp.weight *= scaleFactor

    def _aggregate_expert_predictions(self, predictions):
        """
        Aggregate predictions of all experts according to their weights.
        Returns array of shape: (n_classes,)
        """
        aggregate_preds = np.zeros((np.max(predictions) + 1,))
        for pred, w in zip(predictions, (exp.weight for exp in self.experts)):
            aggregate_preds[pred] += w

        return aggregate_preds / sum(exp.weight for exp in self.experts)

    def _remove_experts(self):
        """
        Removes all experts whose weight is lower than self.theta.
        """
        self.experts = [exp for exp in self.experts if exp.weight >= self.theta]

    def _construct_new_expert(self):
        """
        Constructs a new WeightedExpert from the provided base_estimator.
        """
        return self.WeightedExpert(cp.deepcopy(self.base_estimator), 1)

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return \
            type(self).__name__ + ': ' + \
            "max_estimators: {} - ".format(self.max_experts) + \
            "base_estimator: {} - ".format(self.base_estimator.get_info()) + \
            "period: {} - ".format(self.period) + \
            "beta: {} - ".format(self.beta) + \
            "theta: {}".format(self.theta)
