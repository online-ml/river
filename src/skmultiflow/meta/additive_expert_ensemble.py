from skmultiflow.core.base import StreamModel
import numpy as np


class AdditiveExpertEnsemble(StreamModel):
    """
    Additive Expert Ensemble [1].

    Parameters
    __________
    max_estimators: maximum number of estimators to hold.
    base_estimator: constructor for new estimators.
    beta: factor for decreasing weights.
    gamma: factor for new expert weight.

    References
    __________
    [1]
    Kolter and Maloof, Using additive expert ensembles to cope with Concept
    drift, Proc. 22 International Conference on Machine Learning, 2005.
    """

    class WeightedClassifier:
        """
        Wrapper that includes an estimator and its weight, for easier ordering.
        - Inspired by PR #97
        """
        def __init__(self, estimator, weight):
            self.estimator = estimator
            self.weight = weight

        def __lt__(self, other):
            self.weight < other.weight

    def __init__(self, max_estimators, base_estimator, beta, gamma):
        super().__init__()

        self.max_estimators = max_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.gamma = gamma

        self.reset()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError        

    def partial_fit(self, X, y, classes=None, weight=None):
        pass

    def predict(self, X):
        return np.argmax(self.predict_proba(X))

    def predict_proba(self, X):
        predictions = [exp.estimator.predict(X) * exp.weight for exp in self.experts]
        import ipdb; ipdb.set_trace()
        return np.sum(predictions, axis=0) / np.sum(predictions)

    def fit_single_sample(self, x, y, classes=None, weight=None):
        """
        Predict + update weights + modify experts + train on new sample.
        (As was originally described by [1])
        """
        assert len(x.shape) == 1
        import ipdb; ipdb.set_trace()

        ## 1. Get expert predictions:
        predictions_probs = self.get_expert_predictions_probs([x])

        ## 2. Output prediction:
        output_pred = np.argmax(np.sum(
            [pred_probs * w for pred_probs, w in zip(predictions_probs, (exp.weight for exp in self.experts))],
            axis=0
        ))

        ## 3. Update expert weights:
        self.update_expert_weights(map(np.argmax, predictions_probs), y)

        ## 4. If y_pred != y_true, then add a new expert:
        ## new expert's weight is equal to the total weight of the ensemble times the gamma constant
        if output_pred != y:
            ensemble_weight = sum(exp.weight for exp in self.experts)
            new_exp = self.WeightedClassifier(self.base_estimator(), ensemble_weight * self.gamma)

        ## 5. Train each expert on X
        for exp in self.experts:
            exp.estimator.partial_fit(X, y)

        ## 6. Pruning
        ## TODO Pruning to max_estimators
        ## (either by age or weight)

        ## TODO Improve efficieny
        ## there are lots of repeated iterations and O(n) accesses to collections...

    def get_expert_predictions_probs(self, X):
        """
        Returns prediction probabilities of each class for each expert.
        In shape: (rows, cols) = (n_experts, n_classes)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def update_expert_weights(self, expert_predictions, y_true):
        for exp, y_pred in zip(self.experts, expert_predictions):
            if y_pred != y_true:
                exp.weight = exp.weight * self.beta
        
        self.experts = sorted(self.experts, key=lambda exp: exp.weight, reverse=True)
    
    def reset(self):
        self.experts = [self.WeightedClassifier(self.base_estimator(), 1)]

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        pass