import abc
import copy
import math
import random
import typing

from river import base
from river import compose
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing
from river import utils


__all__ = [
    'EpsilonGreedyRegressor',
    'UCBRegressor',
]


class Bandit(base.EnsembleMixin):

    def __init__(self, models: typing.List[base.Estimator], metric: metrics.Metric, explore_each_arm: int, start_after: int, seed: int = None):

        if len(models) <= 1:
            raise ValueError(f"You supply {len(models)} models. At least 2 models should be supplied.")

        # Check that the model and the metric are in accordance
        for model in models:
            if not metric.works_with(model):
                raise ValueError(f"{metric.__class__.__name__} metric can't be used to evaluate a " +
                                 f'{model.__class__.__name__}')
        super().__init__(models)
        self.metric = copy.deepcopy(metric)
        self._y_scaler = copy.deepcopy(preprocessing.StandardScaler())

        # Initializing bandits internals
        self._n_arms = len(models)
        self._n_iter = 0 # number of times learn_one is called
        self._N = [0] * self._n_arms
        self.explore_each_arm = explore_each_arm
        self._average_reward = [0.0] * self._n_arms

        # Warm up
        self.start_after = start_after
        self.warm_up = True

        # Randomization
        self.seed = seed
        self._rng = random.Random(seed)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}" +
            f"\n\t{str(self.metric)}" +
            f"\n\t{'Best model id: ' + str(self._best_model_idx)}"
        ).expandtabs(2)

    @abc.abstractmethod
    def _pull_arm(self):
        pass

    @abc.abstractmethod
    def _update_arm(self, arm, reward):
        pass

    @abc.abstractmethod
    def _pred_func(self, model):
        pass

    @property
    def _best_model_idx(self):
        # average reward instead of cumulated (otherwise favors arms which are pulled often)
        return utils.math.argmax(self._average_reward)

    @property
    def best_model(self):
        return self[self._best_model_idx]

    @property
    def percentage_pulled(self):
        if not self.warm_up:
            percentages = [n / sum(self._N) for n in self._N]
        else:
            percentages = [0] * self._n_arms
        return percentages

    def predict_one(self, x):
        best_arm = self._best_model_idx
        y_pred = self._pred_func(self[best_arm])(x)

        return y_pred

    def learn_one(self, x, y):
        self._learn_one(x, y)
        return self

    def add_models(self, new_models: typing.List[base.Estimator]):
        length_new_models = len(new_models)
        self.models += new_models
        self._n_arms += length_new_models
        self._N += [0] * length_new_models
        self._average_reward += [0.0] * length_new_models

    def _learn_one(self, x, y):
        # Explore all arms pulled less than `explore_each_arm` times
        never_pulled_arm = [i for (i, n) in enumerate(self._N) if n < self.explore_each_arm]
        if never_pulled_arm:
            chosen_arm = self._rng.choice(never_pulled_arm)
        else:
            chosen_arm = self._pull_arm()

        # Predict and learn with the chosen model
        chosen_model = self[chosen_arm]
        y_pred = chosen_model.predict_one(x)
        self.metric.update(y_pred=y_pred, y_true=y)
        chosen_model.learn_one(x=x, y=y)

        # Update bandit internals
        if self.warm_up and (self._n_iter == self.start_after):
            self._n_iter = 0 # must be reset to 0 since it is an input to some model
            self.warm_up = False

        self._n_iter += 1
        reward = self._compute_scaled_reward(y_pred=y_pred, y_true=y)
        if not self.warm_up:
            self._update_bandit(chosen_arm=chosen_arm, reward=reward)

        return self.metric._eval(y_pred, y), reward, chosen_arm

    def _update_bandit(self, chosen_arm, reward):
        # Updates common to all bandits
        self._n_iter += 1
        self._N[chosen_arm] += 1
        self._average_reward[chosen_arm] += (1.0 / self._N[chosen_arm]) * \
                                            (reward - self._average_reward[chosen_arm])

        # Specific update of the arm for certain bandit model
        self._update_arm(chosen_arm, reward)


    def _compute_scaled_reward(self, y_pred, y_true, c=1, scale_y=True):
        if scale_y:
            y_true = self._y_scaler.learn_one(dict(y=y_true)).transform_one(dict(y=y_true))["y"]
            y_pred = self._y_scaler.transform_one(dict(y=y_pred))["y"]

        metric_value = self.metric._eval(y_pred, y_true)
        metric_value = metric_value if self.metric.bigger_is_better else (-1) * metric_value
        reward = 1 / (1 + math.exp(- c * metric_value)) if c * metric_value > -30 else 0 # to avoid overflow

        return reward


class EpsilonGreedyBandit(Bandit):

    def __init__(self, models: typing.List[base.Estimator], metric: metrics.Metric, seed: int = None, start_after=25,
                 epsilon=0.1, epsilon_decay=None, explore_each_arm=0):
        super().__init__(models=models, metric=metric, seed=seed, explore_each_arm=explore_each_arm, start_after=start_after)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if epsilon_decay:
            self._starting_epsilon = epsilon

    def _pull_arm(self):
        if self._rng.random() > self.epsilon:
            chosen_arm = utils.math.argmax(self._average_reward)
        else:
            chosen_arm = self._rng.choice(range(self._n_arms))

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm internals are already updated in the `learn_one` phase of class `Bandit`.
        if self.epsilon_decay:
            self.epsilon = self._starting_epsilon * math.exp(-self._n_iter*self.epsilon_decay)


class EpsilonGreedyRegressor(EpsilonGreedyBandit, base.Regressor):
    """Epsilon-greedy bandit algorithm for regression.

    This bandit selects the best arm (defined as the one with the highest average reward) with
    probability $(1 - \\epsilon)$ and draws a random arm with probability $\\epsilon$. It is also
    called Follow-The-Leader (FTL) algorithm.


    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    epsilon
        Exploration parameter (default : 0.1).


    Examples
    --------
    Let's use `UCBRegressor` to select the best learning rate for a linear regression model. First, we define the grid of models:

    >>> from river import compose
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import optim

    >>> models = [
    ...     compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...         ) for lr in [1e-4, 1e-3, 1e-2, 1e-1]
    ... ]

    We decide to use TrumpApproval dataset:

    >>> from river import datasets
    >>> dataset = datasets.TrumpApproval()

    We then define the metric and the scaler for the reward

    >>> from river.expert import EpsilonGreedyRegressor
    >>> from river import metrics

    >>> metric = metrics.MSE()
    >>> bandit = EpsilonGreedyRegressor(models=models, metric=metric, seed=1)

    We can then train the models in the bandit in an online fashion:




    References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """
    @classmethod
    def _unit_test_params(cls):
        return {
            'models': [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.1)))
            ],
            'metric': metrics.MSE(),
        }

    def _pred_func(self, model):
        return model.predict_one


class UCBBandit(Bandit):

    def __init__(self, models: typing.List[base.Estimator], metric: metrics.Metric, seed: int = None, start_after=25, explore_each_arm=1, delta=None):
        if explore_each_arm < 1:
            raise ValueError("Argument 'explore_each_arm' should be >= 1")
        super().__init__(models=models, metric=metric, seed=seed, explore_each_arm=explore_each_arm, start_after=start_after)
        if delta is not None and (delta >= 1 or delta <= 0):
            raise ValueError("The parameter delta should be comprised in ]0, 1[ (or set to None)")
        self.delta = delta

    def _pull_arm(self):
        if self.delta:
            exploration_bonus = [math.sqrt(2 * math.log(1/self.delta) / n) for n in self._N]
        else:
            exploration_bonus = [math.sqrt(2 * math.log(self._n_iter) / n) for n in self._N]
        upper_bound = [
            avg_reward + exploration
            for (avg_reward, exploration)
            in zip(self._average_reward, exploration_bonus)
        ]
        chosen_arm = utils.math.argmax(upper_bound)

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm internals are already updated in the `learn_one` phase of class `Bandit`.
        pass


class UCBRegressor(UCBBandit, base.Regressor):
    """Upper Confidence Bound bandit for regression.

    The class offers 2 implementations of UCB:
    - UCB1 from [^1], when the parameter delta has value None
    - UCB(delta) from [^2], when the parameter delta is in (0, 1)

    For this bandit, rewards are supposed to be 1-subgaussian (see Lattimore and Szepesvári,
    chapter 6, p. 91) hence the use of the `StandardScaler` and `MaxAbsScaler` as `reward_scaler`.

    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    delta
        For UCB(delta) implementation. Lower value means more exploration.

     Let's use `UCBRegressor` to select the best learning rate for a linear regression model.

    Examples
    --------
    Let's use `UCBRegressor` to select the best learning rate for a linear regression model. First, we define the grid of models:

    >>> from river import compose
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import optim

    >>> models = [
    ...     compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...         ) for lr in [1e-4, 1e-3, 1e-2, 1e-1]
    ... ]

    We decide to use TrumpApproval dataset:

    >>> from river import datasets
    >>> dataset = datasets.TrumpApproval()

    We then define the metric

    >>> from river.expert import UCBRegressor
    >>> from river import metrics

    >>> metric = metrics.MSE()
    >>> bandit = UCBRegressor(models=models, metric=metric, seed=1)

    We can then train the models in the bandit in an online fashion:


    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    [^2]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    [^3]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    """
    @classmethod
    def _unit_test_params(cls):
        return {
            'models': [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01))),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.1)))
            ],
            'metric': metrics.MSE(),
        }

    def _pred_func(self, model):
        return model.predict_one
