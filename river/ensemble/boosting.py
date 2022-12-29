import collections
import math
import typing

from river import base, drift, linear_model, utils

__all__ = ["AdaBoostClassifier", "BOLEClassifier"]


class AdaBoostClassifier(base.WrapperEnsemble, base.Classifier):
    """Boosting for classification

    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter lambda. The lambda parameter is
    updated when the weaks learners fit successively the same observation.

    Parameters
    ----------
    model
        The classifier to boost.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.

    Attributes
    ----------
    wrong_weight : collections.defaultdict
        Number of times a model has made a mistake when making predictions.
    correct_weight : collections.defaultdict
        Number of times a model has predicted the right label when making predictions.

    Examples
    --------

    In the following example three tree classifiers are boosted together. The performance is
    slightly better than when using a single tree.

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Phishing()

    >>> metric = metrics.LogLoss()

    >>> model = ensemble.AdaBoostClassifier(
    ...     model=(
    ...         tree.HoeffdingTreeClassifier(
    ...             split_criterion='gini',
    ...             delta=1e-5,
    ...             grace_period=2000
    ...         )
    ...     ),
    ...     n_models=5,
    ...     seed=42
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.370805

    >>> print(model)
    AdaBoostClassifier(HoeffdingTreeClassifier)

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None):
        super().__init__(model, n_models, seed)
        self.wrong_weight: typing.DefaultDict = collections.defaultdict(int)
        self.correct_weight: typing.DefaultDict = collections.defaultdict(int)

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LogisticRegression()}

    def learn_one(self, x, y):
        lambda_poisson = 1

        for i, model in enumerate(self):
            for _ in range(utils.random.poisson(lambda_poisson, self._rng)):
                model.learn_one(x, y)

            if model.predict_one(x) == y:
                self.correct_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.correct_weight[i]
                )

            else:
                self.wrong_weight[i] += lambda_poisson
                lambda_poisson *= (self.correct_weight[i] + self.wrong_weight[i]) / (
                    2 * self.wrong_weight[i]
                )
        return self

    def predict_proba_one(self, x):
        y_proba = collections.Counter()

        for i, model in enumerate(self):
            epsilon = self.wrong_weight[i] + 1e-16
            epsilon /= (self.correct_weight[i] + self.wrong_weight[i]) + 1e-16
            if epsilon == 0 or epsilon > 0.5:
                model_weight = 1.0
            else:
                beta_inv = (1 - epsilon) / epsilon
                model_weight = math.log(beta_inv) if beta_inv != 0 else 0
            predictions = model.predict_proba_one(x)
            utils.norm.normalize_values_in_dict(predictions, inplace=True)
            utils.norm.scale_values_in_dict(predictions, model_weight, inplace=True)
            y_proba.update(predictions)

        utils.norm.normalize_values_in_dict(y_proba, inplace=True)
        return y_proba


class BOLEClassifier(AdaBoostClassifier):
    """Boosting Online Learning Ensemble (BOLE)

    A modified version of Oza Online Boosting Algorithm [^1]. For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter lambda. The first model to be trained will be the one with worst correct_weight / (correct_weight + wrong_weight).
    The worst model's not yet trained will receive lambda values for training from the model's that incorrectly classified an instance, and the best model's not yet trained
    will receive lambda values for training from the model's that correctly classified an instance. For more details, see [^2].

    Parameters
    ----------
    model
        The classifier to boost.
    n_models
        The number of models in the ensemble.
    seed
        Random number generator seed for reproducibility.
    error_bound
        Error bound percentage for allowing models to vote.

    Attributes
    ----------
    wrong_weight : collections.defaultdict
        Number of times a model has made a mistake when making predictions.
    correct_weight : collections.defaultdict
        Number of times a model has predicted the right label when making predictions.
    order_position :
        Array with the index of the models with best (correct_weight / correct_weight + wrong_weight) in descending order.
    instances_seen :
        Number of instances that the ensemble trained with.

    Examples
    --------

    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import drift
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Elec2().take(3000)

    >>> model = ensemble.BOLEClassifier(
    ...     model=drift.DriftRetrainingClassifier(
    ...         model=tree.HoeffdingTreeClassifier(),
    ...         drift_detector=drift.DDM()
    ...     ),
    ...     n_models=10,
    ...     seed=42
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 93.63%

    References
    ----------
    [^1]: [Oza, N.C., 2005, October. Online bagging and boosting. In 2005 IEEE international conference on systems, man and cybernetics (Vol. 3, pp. 2340-2345). Ieee.](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
    [^2]: R. S. M. d. Barros, S. Garrido T. de Carvalho Santos and P. M. Gonçalves Júnior, "A Boosting-like Online Learning Ensemble," 2016 International Joint Conference on Neural Networks (IJCNN), 2016, pp. 1871-1878, doi: 10.1109/IJCNN.2016.7727427.

    """

    def __init__(self, model: base.Classifier, n_models=10, seed: int = None, error_bound=0.5):
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.error_bound = error_bound
        self.order_position = [i for i in range(n_models)]
        self.instances_seen = 0

    def learn_one(self, x, y):
        self.instances_seen += 1

        acc = [0] * self.n_models
        for i in range(self.n_models):
            acc[i] = (
                self.correct_weight[self.order_position[i]]
                + self.wrong_weight[self.order_position[i]]
            )
            if acc[i] != 0:
                acc[i] = self.correct_weight[self.order_position[i]] / acc[i]

        # sort models in descending order by correct_weight / correct_weight + error_weight (best case insertion sort)
        for i in range(1, self.n_models):
            key_position = self.order_position[i]
            key_acc = acc[i]
            j = i - 1
            while (j >= 0) and (acc[j] < key_acc):
                self.order_position[j + 1] = self.order_position[j]
                acc[j + 1] = acc[j]
                j -= 1
            self.order_position[j + 1] = key_position
            acc[j + 1] = key_acc

        correct = False
        max_acc = 0
        min_acc = self.n_models - 1
        lambda_poisson = 1

        # does boosting not in a linear way (i.e a model boosts the model created after him in the ensemble creation process)
        # the first model to be trained will be the one with worst correct_weight / correct_weight + wrong_weight
        # The worst model's not yet trained will receive lambda values for training from the model's that incorrectly classified an instance.
        # the best model's not yet trained will receive lambda values for training from the model's that correctly classified an instance.
        # the values of lambda increase in case a mistake is made and decrease in case a right prediction is made.
        # the worst models are more likely to make mistakes, increasing the value of lambda.
        # Then, the best's model are likely to receive a high value of lambda and decreasing gradually throughout the remaning models to be trained
        # It's similar to a system where the rich get richer.
        for i in range(self.n_models):
            if correct:
                pos = self.order_position[max_acc]
                max_acc += 1
            else:
                pos = self.order_position[min_acc]
                min_acc -= 1

            for _ in range(utils.random.poisson(lambda_poisson, self._rng)):
                self.models[pos].learn_one(x, y)

            if self.models[pos].predict_one(x) == y:
                self.correct_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.correct_weight[pos])
                correct = True
            else:
                self.wrong_weight[pos] += lambda_poisson
                lambda_poisson *= (self.instances_seen) / (2 * self.wrong_weight[pos])
                correct = False
        return self

    def predict_proba_one(self, x):
        y_proba = collections.Counter()
        for i, model in enumerate(self):
            model_weight = 0.0
            if self.correct_weight[i] > 0.0 and self.wrong_weight[i] > 0.0:
                epsilon = self.wrong_weight[i] / (self.correct_weight[i] + self.wrong_weight[i])
                if epsilon <= self.error_bound:
                    beta_inv = (1 - epsilon) / epsilon
                    model_weight = math.log(beta_inv)

            if model_weight:
                predictions = model.predict_proba_one(x)
                utils.norm.normalize_values_in_dict(predictions, inplace=True)
                utils.norm.scale_values_in_dict(predictions, model_weight, inplace=True)
                y_proba.update(predictions)

        utils.norm.normalize_values_in_dict(y_proba, inplace=True)
        return y_proba
