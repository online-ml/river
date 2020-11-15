import abc

import numpy as np

from river import metrics


__all__ = [
    'EpsilonGreedyBandit',
    'UCBBandit',
    'RandomBandit',
    'OracleBandit',
    'Exp3Bandit'
]

# TODO :
# type annotation in __init__
# loss-based reward ('compute_reward')
# tests for classification
# tests on real datasets


class Bandit(metaclass=abc.ABCMeta):
    """Abstract class for bandit.

    References
    ----------
    [^1] [Python code from Toulouse univ](https://www.math.univ-toulouse.fr/~jlouedec/demoBandits.html)
    [^2] [Slivkins, A. Introduction to Multi-Armed Bandits](https://arxiv.org/pdf/1904.07272.pdf)
    """

    def __init__(self, models, compute_reward, metric: metrics.Metric = None,
                 verbose=False, save_rewards=False):
        if len(models) <= 1:
            raise ValueError(f"You supply {len(models)} models. At least 2 models should be supplied.")
        
        self.compute_reward = compute_reward
        self.metric = metric
        self.verbose = verbose
        self.models = models
        self.save_rewards = save_rewards
        if save_rewards:
            self.rewards = []

        self._n_arms = len(models)
        self._Q = np.zeros(self._n_arms, dtype=np.float)
        self._N = np.ones(self._n_arms, dtype=np.int)

    @abc.abstractmethod
    def _pull_arm(self):
        pass

    @abc.abstractmethod
    def _update_arm(self, arm, reward):
        pass

    @property
    def _best_model_idx(self):
        return np.argmax(self._Q)

    @property
    def best_model(self):
        return self.models[self._best_model_idx]

    def print_percentage_pulled(self):
        percentages = self._N / sum(self._N)
        for i, pct in enumerate(percentages):
            print(f"Model {i}: {round(pct*100, 2)}% of the time")

    def predict_one(self, x):
        best_arm = self._pull_arm()
        y_pred = self.models[best_arm].predict_one(x)

        return y_pred

    def learn_one(self, x, y):
        best_arm = self._pull_arm()
        best_model = self.models[best_arm]
        y_pred = best_model.predict_one(x)

        best_model.learn_one(x=x, y=y)

        if self.verbose:
            print(
                '\t'.join((
                    f'best {self._best_model_idx}',
                ))
            )
            print("y_pred:", y_pred, ", y_true:", y)

        reward = self.compute_reward(y_pred=y_pred, y_true=y)

        if self.metric:
            self.metric.update(y_pred=y_pred, y_true=y)

        if self.save_rewards:
            self.rewards += [reward]

        self._update_arm(best_arm, reward)

        return self


class EpsilonGreedyBandit(Bandit):
    """Epsilon-greedy bandit implementation
    
    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    epsilon
        Exploration parameter (default : 0.1).
    reduce_epsilon
        Factor applied to reduce epsilon at each time-step (default : 0.99).


    Examples
    --------
    TODO

     References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    """

    def __init__(self, epsilon=0.1, reduce_epsilon=0.99, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.reduce_epsilon = reduce_epsilon

    def _pull_arm(self):
        if np.random.rand() > self.epsilon:
            best_arm = np.argmax(self._Q)
        else:
            best_arm = np.random.choice(self._n_arms)

        return best_arm

    def _update_arm(self, arm, reward):
        if self.reduce_epsilon:
            self.epsilon *= self.reduce_epsilon
        self._N[arm] += 1
        self._Q[arm] += (1.0 / self._N[arm]) * (reward - self._Q[arm])


class UCBBandit(Bandit):
    """Upper Confidence Bound bandit.
    
     References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    """

    def __init__(self, delta, **kwargs):
        super().__init__(**kwargs)
        self._t = 1
        self.delta = delta

    def _pull_arm(self):
        #upper_bound = self._Q + np.sqrt(2*np.log(self._t)/self._N)
        upper_bound = self._Q + np.sqrt(2*np.log(1/self.delta)/self._N)
        best_arm = np.argmax(upper_bound)
        return best_arm

    def _update_arm(self, arm, reward):
        self._N[arm] += 1
        self._Q[arm] += (1.0 / self._N[arm]) * (reward - self._Q[arm])
        self._t += 1


class Exp3Bandit(Bandit):
    """Exp3 implementation from Lattimore and Szepesvári. The algorithm makes the hypothesis that the reward is in [0, 1]

    References
    ----------
    [^1]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    """

    def __init__(self, gamma=0.5, **kwargs):
        super().__init__(**kwargs)
        self._p = np.ones(self._n_arms, dtype=np.float)
        self._s = np.zeros(self._n_arms, dtype=np.float)
        self.gamma = gamma

    def _make_distr(self):
        numerator =  np.exp(self.gamma * self._s)
        return numerator / np.sum(numerator)

    def _pull_arm(self):
        self._p = self._make_distr()
        best_arm = np.random.choice(a=range(self._n_arms), size=1, p=self._p)[0]
        return best_arm

    def _update_arm(self, arm, reward):
        self._s += 1.0
        self._s[arm]-= (1.0 - reward)/self._p[arm]


class RandomBandit(Bandit):
    """Bandit that does random selection and update of models.
    It is just created for testing purpose"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _pull_arm(self):
        random_arm = np.random.choice(len(self._N))
        return random_arm

    def _update_arm(self, arm, reward):
        pass


class OracleBandit(Bandit):
    """Oracle bandit that draws and updates every models.
    The `predict_one` is the prediction that minimize the error for this time step.
    The `update_one` update all the models available."""

    def __init__(self, save_predictions=True, **kwargs):
        super().__init__(**kwargs)
        self.save_predictions = save_predictions
        if save_predictions:
            self.predictions = []

    @property
    def _best_model_idx(self):
        cum_rewards = np.max(np.array(self.rewards), axis=0)
        return np.argmax(cum_rewards)

    @property
    def max_rewards(self):
        return np.max(np.array(self.rewards), axis=1)

    def predict_one(self, x, y):
        best_arm = self._pull_arm(x, y)
        y_pred = self.models[best_arm].predict_one(x)
        return y_pred

    def _pull_arm(self, x, y):
        preds = [model.predict_one(x) for model in self.models]
        losses = [np.abs(y_pred - y) for y_pred in preds]
        best_arm = np.argmin(losses)
        return best_arm

    def _update_arm(self, arm, reward):
        pass

    def learn_one(self, x, y):
        reward = []
        prediction = []
        for model in self.models:
            y_pred = model.predict_one(x=x)
            prediction += [y_pred]
            r = self.compute_reward(y_pred=y_pred, y_true=y)
            reward += [r]
            model.learn_one(x=x, y=y)

        if self.save_predictions:
            self.predictions += [prediction]
        if self.save_rewards:
            self.rewards += [reward]
