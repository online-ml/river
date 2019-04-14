import copy as cp
import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.bayes import NaiveBayes


class AdditiveExpertEnsemble(StreamModel):
    """
    Additive Expert Ensemble [1].

    Parameters
    __________
    max_estimators: maximum number of estimators to hold.
    base_estimator: constructor for new estimators.
    beta: factor for decreasing weights.
    gamma: factor for new expert weight.
    n_pretrain_samples: number of samples to train on before starting to
        update expert weights.

    References
    __________
    .. [1] Kolter and Maloof. Using additive expert ensembles to cope with
        Concept drift. Proc. 22 International Conference on Machine Learning,
        2005.
    """

    class WeightedClassifier:
        """
        Wrapper that includes an estimator and its weight, for easier ordering.
        """
        def __init__(self, estimator, weight):
            self.estimator = estimator
            self.weight = weight

        def __lt__(self, other):
            return self.weight < other.weight

    def __init__(self, max_estimators, base_estimator=NaiveBayes(),
                 beta=0.9, gamma=0.1, n_pretrain_samples=50):
        super().__init__()

        self.max_estimators = max_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.gamma = gamma
        self.n_pretrain_samples = n_pretrain_samples

        self.reset()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        if self.n_samples < self.n_pretrain_samples:
            for exp in self.experts:
                exp.estimator.partial_fit(X, y, classes=classes, weight=weight)
        else:
            for i in range(len(X)):
                self.fit_single_sample(
                    X[i:i+1, :], y[i:i+1], classes, weight
                )

        self.n_samples += X.shape[0]

    def predict(self, X):
        return np.array([np.argmax(self.predict_proba(X))])

    def predict_proba(self, X):
        return self._aggregate_expert_predictions(
                    self.get_expert_predictions(X))

    def fit_single_sample(self, X, y, classes=None, weight=None):
        """
        Predict + update weights + modify experts + train on new sample.
        (As was originally described by [1])
        """
        # 1. Get expert predictions:
        predictions = self.get_expert_predictions(X)

        # 2. Get aggregate prediction:
        output_pred = np.argmax(
                        self._aggregate_expert_predictions(predictions))

        # 3. Update expert weights:
        self.update_expert_weights(predictions, y)

        # 4. If y_pred != y_true, then add a new expert:
        if output_pred != np.asscalar(y):
            ensemble_weight = sum(exp.weight for exp in self.experts)
            new_exp = self.WeightedClassifier(
                        self._construct_base_estimator(),
                        ensemble_weight * self.gamma)
            self._add_expert(new_exp)

        # 4.1 Pruning to self.max_estimators if needed
        if len(self.experts) > self.max_estimators:
            self.experts.pop()

        # 5. Train each expert on X
        for exp in self.experts:
            exp.estimator.partial_fit(X, y, classes=classes, weight=weight)

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts,)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def _aggregate_expert_predictions(self, predictions):
        """
        Aggregate predictions of all experts according to their weights.
        Returns array of shape: (n_classes,)
        """
        aggregate_preds = np.zeros((np.max(predictions) + 1,))
        for pred, w in zip(predictions, (exp.weight for exp in self.experts)):
            aggregate_preds[pred] += w

        return aggregate_preds / sum(exp.weight for exp in self.experts)

    def _add_expert(self, new_exp):
        """
        Inserts the new expert on the sorted self.experts list.
        """
        idx = 0
        for exp in self.experts:
            if exp.weight < new_exp.weight:
                break
            idx += 1
        self.experts.insert(idx, new_exp)

    def _construct_base_estimator(self):
        return cp.deepcopy(self.base_estimator)

    def update_expert_weights(self, expert_predictions, y_true):
        for exp, y_pred in zip(self.experts, expert_predictions):
            if np.all(y_pred == y_true):
                continue
            exp.weight = exp.weight * self.beta

        self.experts = sorted(self.experts, key=lambda exp: exp.weight,
                              reverse=True)

    def reset(self):
        self.n_samples = 0
        self.experts = [
            self.WeightedClassifier(self._construct_base_estimator(), 1)
        ]

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        return \
            type(self).__name__ + ': ' + \
            "max_estimators: {} - ".format(self.max_estimators) + \
            "base_estimator: {} - ".format(self.base_estimator.get_info()) + \
            "beta: {} - ".format(self.beta) + \
            "gamma: {}".format(self.gamma)
