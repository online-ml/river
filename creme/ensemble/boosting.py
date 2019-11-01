import collections
import copy

import numpy as np

from sklearn import utils

from .. import base


__all__ = ['AdaboostClassifier']


class BaseBoosting(base.Wrapper, base.Ensemble):

    def __init__(self, model, n_models=10, random_state=None):
        super().__init__(copy.deepcopy(model) for i in range(n_models))
        self.model = model
        self.rng = utils.check_random_state(random_state)

    @property
    def _model(self):
        return self.model


class AdaboostClassifier(BaseBoosting):
    '''Boosting for classification

    For each incoming observation, each model's ``fit_one`` method is called ``k`` times where
    ``k`` is sampled from a Poisson distribution of parameter lambda. The lambda parameter is 
    updated when the weaks learners fit successively the same observation. 

    Parameters:
        model (BinaryClassifier or MultiClassifier): The classifier to boost.
        n_models (int): The number of models in the ensemble.
        random_state (int, ``numpy.random.RandomState`` instance or None): If int, ``random_state``
            is the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by `numpy.random`.

    Example:

        In the following example three tree classifiers are bagged together. The performance is
        slightly better than when using a single tree.

        ::
            >>> from creme import datasets
            >>> from creme import ensemble
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import tree

            >>> X_y = datasets.fetch_electricity()

            >>> metric = metrics.LogLoss()

            >>> model = ensemble.AdaboostClassifier(
            ...     model=(
            ...         tree.DecisionTreeClassifier(
            ...             criterion='gini', 
            ...             confidence=1e-5, 
            ...             patience=2000
            ...         )
            ...     ),
            ...     n_models=5,
            ...     random_state=42
            ... )

            >>> model_selection.online_score(X_y, model, metric)
            LogLoss: 0.5531 

            >>> print(model)
            AdaboostClassifier(DecisionTreeClassifier)

    References:
        1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_
        2. https://github.com/crm416/online_boosting/blob/master/ensemblers/adaboost.py

    '''

    def __init__(self, model, n_models=10, random_state=None):
        super().__init__(model, n_models, random_state)
        self.wrong_weight   = collections.defaultdict(int)
        self.correct_weight = collections.defaultdict(int)

    def fit_one(self, x, y):  
        lambda_poisson = 1
        
        for i, model in enumerate(self):
            for _ in range(self.rng.poisson(lambda_poisson)):
                model.fit_one(x, y)
            
            if model.predict_one(x) == y:
                self.correct_weight[i] += lambda_poisson
                lambda_poisson *= (
                    (self.correct_weight[i] + self.wrong_weight[i]) / (2 * self.correct_weight[i]) 
                )
            
            else:
                self.wrong_weight[i] += lambda_poisson
                lambda_poisson *= (
                    (self.correct_weight[i] + self.wrong_weight[i]) / (2 * self.wrong_weight[i])
                )
        return self


    def predict_proba_one(self, x):
        """
        Store the predicted probabilities with the corresponding weights for each weak learner and 
        return the probabilities associated to the model which has maximum weight.
        """
        model_weights = collections.defaultdict(int)
        predictions   = {}

        for i, model in enumerate(self):
            epsilon  = self.correct_weight[i] + 1e-16
            epsilon /= (self.wrong_weight[i]  + 1e-16)
            weight   = np.log(epsilon)
            model_weights[i] += weight 
            predictions[i] = model.predict_proba_one(x)

        y_pred = predictions[max(model_weights, key=model_weights.get)]
        total = sum(y_pred.values())
        return {label: proba / total for label, proba in y_pred.items()}


    def predict_one(self, x):
        """
        Store the predictions with the corresponding weights for each weak learner and return the 
        prediction associated with the maximum weight.
        """
        return self.predict_proba_one(x)

