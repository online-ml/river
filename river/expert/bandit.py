import abc
import copy
import math
import random
import typing

from river import base
from river import metrics
from river import preprocessing


__all__ = [
    'EpsilonGreedyRegressor',
    'UCBRegressor',
]

# TODO:
# Docstring
# Determine which object to store (rewards/percentages pulled/loss?)

def argmax(l):
    return max(range(len(l)), key=l.__getitem__)

class Bandit(base.EnsembleMixin):

    def __init__(self, models, metric: metrics.Metric, reward_scaler: base.Transformer):

        if len(models) <= 1:
            raise ValueError(f"You supply {len(models)} models. At least 2 models should be supplied.")

        # Check that the model and the metric are in accordance
        for model in models:
            if not metric.works_with(model):
                raise ValueError(f"{metric.__class__.__name__} metric can't be used to evaluate a " +
                                 f'{model.__class__.__name__}')
        super().__init__(models)
        self.reward_scaler = copy.deepcopy(reward_scaler)
        self.metric = copy.deepcopy(metric)

        # Initializing bandits internals
        self._n_arms = len(models)
        self._n_iter = 0 # number of times learn_one is called
        self._N = [0] * self._n_arms
        self._average_reward = [0.0] * self._n_arms

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
        return argmax(self._average_reward)

    @property
    def best_model(self):
        return self[self._best_model_idx]

    @property
    def percentage_pulled(self):
        percentages = [n / sum(self._N) for n in self._N]
        return percentages

    def predict_one(self, x):
        best_arm = self._pull_arm()
        y_pred = self._pred_func(self[best_arm])(x)
        return y_pred

    def learn_one(self, x, y):
        self._learn_one(x, y)
        return self

    def add_models(self, new_models):
        if not isinstance(new_models, list):
            raise TypeError("Argument `new_models` must be of a list")

        length_new_models = len(new_models)
        # Careful, not validation of the model is done here (contrary to __init__)
        self.models += new_models
        self._n_arms += length_new_models
        self._N = self._N + [0] * length_new_models
        self._average_reward = self._average_reward + [0.0] * length_new_models

    def _learn_one(self, x, y):
        chosen_arm = self._pull_arm()
        chosen_model = self[chosen_arm]

        y_pred = chosen_model.predict_one(x)
        self.metric.update(y_pred=y_pred, y_true=y)
        chosen_model.learn_one(x=x, y=y)

        # Update bandit internals (common to all bandit)
        reward = self._compute_scaled_reward(y_pred=y_pred, y_true=y)
        self._n_iter += 1
        self._N[chosen_arm] += 1
        self._average_reward[chosen_arm] += (1.0 / self._N[chosen_arm]) * \
                                            (reward - self._average_reward[chosen_arm])

        # Specific update of the arm for certain bandit class
        self._update_arm(chosen_arm, reward)

        return self, self.percentage_pulled, self.metric._eval(y_pred, y)

    def _compute_scaled_reward(self, y_pred, y_true, update_scaler=True):
        metric_value = self.metric._eval(y_pred, y_true)
        metric_to_reward_dict = {
            "metric": metric_value if self.metric.bigger_is_better else (-1) * metric_value
        }
        if update_scaler:
            self.reward_scaler.learn_one(metric_to_reward_dict)
        reward = self.reward_scaler.transform_one(metric_to_reward_dict)["metric"]
        return reward


class EpsilonGreedyBandit(Bandit):

    def __init__(self,  models, metric: metrics.Metric, reward_scaler: base.Transformer,
                 epsilon=0.1, epsilon_decay=None):
        super().__init__(models=models, metric=metric, reward_scaler=reward_scaler)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if epsilon_decay:
            self._starting_epsilon = epsilon
        if not self.reward_scaler:
            self.reward_scaler = preprocessing.StandardScaler()

    def _pull_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = argmax(self._average_reward)
        else:
            chosen_arm = random.choice(range(self._n_arms))

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm internals are already updated in the `learn_one` phase of class `Bandit`.
        if self.epsilon_decay:
            self.epsilon = self._starting_epsilon * math.exp(-self._n_iter*self.epsilon_decay)


class EpsilonGreedyRegressor(EpsilonGreedyBandit):
    """Epsilon-greedy bandit algorithm for regression.

    This bandit selects the best arm (defined as the one with the highest average reward) with
    probability $(1 - \\epsilon)$ and draws a random arm with probability $\\epsilon$. It is also
    called Follow-The-Leader (FTL) algorithm.

    For this bandit, reward are supposed to be 1-subgaussian, hence the use of the StandardScaler
    and MaxAbsScaler as `reward_scaler`.

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
    >>> from river import linear_model
    >>> from river import expert
    >>> from river import preprocessing
    >>> from river import metrics


    TODO: finish ex

    References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """

    def _pred_func(self, model):
        return model.predict_one


class UCBBandit(Bandit):

    def __init__(self,  models, metric: metrics.Metric, reward_scaler: base.Transformer,
                 delta=None, explore_each_arm=1):
        super().__init__(models=models, metric=metric, reward_scaler=reward_scaler)
        if delta is not None and (delta >= 1 or delta <= 0):
            raise ValueError("The parameter delta should be comprised in ]0, 1[ (or set to None)")
        self.delta = delta
        self.explore_each_arm = explore_each_arm

        if not self.reward_scaler:
            self.reward_scaler = preprocessing.StandardScaler()

    def _pull_arm(self):
        # Explore all arms pulled less than `explore_each_arm` times
        never_pulled_arm = [i for (i, n) in enumerate(self._N) if n <= self.explore_each_arm]
        if never_pulled_arm:
            chosen_arm = random.choice(never_pulled_arm)
        else:
            if self.delta:
                exploration_bonus = [math.sqrt(2 * math.log(1/self.delta) / n) for n in self._N]
            else:
                exploration_bonus = [math.sqrt(2 * math.log(self._n_iter) / n) for n in self._N]
            upper_bound = [
                avg_reward + exploration
                for (avg_reward, exploration)
                in zip(self._average_reward, exploration_bonus)
            ]
            chosen_arm = argmax(upper_bound)

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


    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    [^2]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    [^3]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    """
    @classmethod
    def _default_params(cls):
        return {'regressor': linear_model.LogisticRegression()}

    def _pred_func(self, model):
        return model.predict_one
