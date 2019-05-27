import copy as cp
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes


class DynamicWeightedMajority(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Dynamic Weighted Majority ensemble classifier.

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
    The dynamic weighted majority (DWM) [1]_, uses four mechanisms to
    cope with concept drift: It trains online learners of the ensemble,
    it weights those learners based on their performance, it removes them,
    also based on their performance, and it adds new experts based on the
    global performance of the ensemble.

    References
    ----------
    .. [1] Kolter and Maloof. Dynamic weighted majority: An ensemble method
       for drifting concepts. The Journal of Machine Learning Research,
       8:2755-2790, December 2007. ISSN 1532-4435.
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

    def __init__(self, n_estimators=5, base_estimator=NaiveBayes(),
                 period=50, beta=0.5, theta=0.01):
        """
        Creates a new instance of DynamicWeightedMajority.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.theta = theta
        self.period = period

        # Following attributes are set later
        self.epochs = None
        self.num_classes = None
        self.experts = None

        self.reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Returns
        -------
        DynamicWeightedMajority
            self
        """
        for i in range(len(X)):
            self.fit_single_sample(
                X[i:i+1, :], y[i:i+1], classes, sample_weight
            )
        return self

    def predict(self, X):
        """ predict

        The predict function will take an average of the predictions of its
        learners, weighted by their respective weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        preds = np.array([np.array(exp.estimator.predict(X)) * exp.weight
                          for exp in self.experts])
        sum_weights = sum(exp.weight for exp in self.experts)
        aggregate = np.sum(preds / sum_weights, axis=0)
        return (aggregate + 0.5).astype(int)    # Round to nearest int

    def predict_proba(self, X):
        raise NotImplementedError

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Fits a single sample of shape `X.shape=(1, n_attributes)` and `y.shape=(1)`

        Aggregates all experts' predictions, diminishes weight of experts whose
        predictions were wrong, and may create or remove experts every _period_
        samples.

        Finally, trains each individual expert on the provided data.

        Train loop as described by Kolter and Maloof in the original paper.


        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: list
            List of all existing classes. This is an optional parameter.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Applicability
            depends on the base estimator.

        """
        self.epochs += 1
        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)
        predictions = np.zeros((self.num_classes,))
        max_weight = 0
        weakest_expert_weight = 1
        weakest_expert_index = None

        for i, exp in enumerate(self.experts):
            y_hat = exp.estimator.predict(X)
            if np.any(y_hat != y) and (self.epochs % self.period == 0):
                exp.weight *= self.beta

            predictions[y_hat] += exp.weight
            max_weight = max(max_weight, exp.weight)

            if exp.weight < weakest_expert_weight:
                weakest_expert_index = i
                weakest_expert_weight = exp.weight

        y_hat = np.array([np.argmax(predictions)])
        if self.epochs % self.period == 0:
            self._scale_weights(max_weight)
            self._remove_experts()
            if np.any(y_hat != y):
                if len(self.experts) == self.n_estimators:
                    self.experts.pop(weakest_expert_index)
                self.experts.append(self._construct_new_expert())

        # Train individual experts
        for exp in self.experts:
            exp.estimator.partial_fit(X, y, classes, sample_weight)

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts, n_samples)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2    # Minimum of 2 classes
        self.experts = [
            self._construct_new_expert()
        ]

    def _scale_weights(self, max_weight):
        """
        Scales the experts' weights such that the max is 1.
        """
        scale_factor = 1 / max_weight
        for exp in self.experts:
            exp.weight *= scale_factor

    def _remove_experts(self):
        """
        Removes all experts whose weight is lower than self.theta.
        """
        self.experts = [ex for ex in self.experts if ex.weight >= self.theta]

    def _construct_new_expert(self):
        """
        Constructs a new WeightedExpert from the provided base_estimator.
        """
        return self.WeightedExpert(cp.deepcopy(self.base_estimator), 1)
