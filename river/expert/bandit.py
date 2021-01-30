import abc
import copy
import math
import random
import typing

from river import base, compose, linear_model, metrics, optim, preprocessing
from river.utils.math import sigmoid

__all__ = ["EpsilonGreedyRegressor", "UCBRegressor"]


def argmax(lst: list, rng: random.Random = None):
    """Argmax function that randomize the returned index in case multiple maxima.

    Mainly used for bandit to avoid exploration bias towards the 1st model in the first iterations.

    Parameters
    ----------
    lst
        The list where the arg max has to be found.
    rng
        The pseudo-random number generator to use.

    """
    max_value = max(lst)
    args_max = [i for (i, x) in enumerate(lst) if x == max_value]
    random_fun = rng.choice if rng else random.choice
    arg_max = random_fun(args_max) if len(args_max) > 0 else args_max[0]

    return arg_max


class Bandit(base.EnsembleMixin):
    """Bandit base class.

    The bandit implementations must inherit from this class and override the abstract method
    `_pull_arm`, `_update_arm` and `_pred_func`.

    Usually, a first subclass inherits from the Bandit class to implement the bandit logic that is
    contained in `_pull_arm` and `_update_arm`. Then, the concrete implementation should inherit
    from said subclass and implement `_pred_func`, according to whether this concrete
    implementation is a regressor or a classifier.

    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    explore_each_arm
        The number of times each arm should explored first.
    start_after
        The number of iteration after which the bandit mechanism should begin.
    seed
        The seed for the algorithm (since not deterministic)."""

    def __init__(
        self,
        models: typing.List[base.Estimator],
        metric: metrics.Metric,
        explore_each_arm: int,
        start_after: int,
        seed: int = None,
    ):

        if len(models) <= 1:
            raise ValueError(
                f"You supplied {len(models)} models. At least 2 models should be supplied."
            )

        # Check that the model and the metric are in accordance
        for model in models:
            if not metric.works_with(model):
                raise ValueError(
                    f"{metric.__class__.__name__} metric can't be used to evaluate a "
                    + f"{model.__class__.__name__}"
                )
        super().__init__(models)
        self.metric = copy.deepcopy(metric)
        self._y_scaler = copy.deepcopy(preprocessing.StandardScaler())

        # Initializing bandits internals
        self._n_arms = len(models)
        self._n_iter = 0  # number of times learn_one is called
        self._N = [0] * self._n_arms
        self.explore_each_arm = explore_each_arm
        self.average_reward = [0.0] * self._n_arms

        # Warm up
        self.start_after = start_after
        self.warm_up = True

        # Randomization
        self.seed = seed
        self._rng = random.Random(seed)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            + f"\n\t{str(self.metric)}"
            + f"\n\t{'Best model id: ' + str(self._best_model_idx)}"
        ).expandtabs(2)

    @abc.abstractmethod
    def _pull_arm(self):
        """Returns the chosen arm.

        It must be overriden in the class that inherits from the Bandit class.

        The way to choose the arm depends on each bandit subclass and expresses the dilemna between
        exploitation and exploration.

        """
        pass

    @abc.abstractmethod
    def _update_arm(self, arm, reward):
        """Update the internals of the chosen arm, based on the value of the reward.

        It must be overriden in the class that inherits from the Bandit class.

        """
        pass

    @abc.abstractmethod
    def _pred_func(self, model):
        pass

    @property
    def _best_model_idx(self):
        """Returns the index of the best model (defined as the one who maximises average reward)."""
        # Average reward instead of cumulated (otherwise favors arms which are pulled often)
        return argmax(self.average_reward, self._rng)

    @property
    def best_model(self):
        """Returns the best model, defined as the one who maximises average reward."""
        return self[self._best_model_idx]

    @property
    def percentage_pulled(self):
        """Returns the number of times (in %) each arm has been pulled."""
        if not self.warm_up:
            return [n / sum(self._N) for n in self._N]

        return [0] * self._n_arms

    def predict_one(self, x):
        """Return the prediction of the best model (defined as the one who maximises average reward)."""
        best_arm = self._best_model_idx
        y_pred = self._pred_func(self[best_arm])(x)

        return y_pred

    def learn_one(self, x, y):
        """Updates the chosen model and the arm internals (the actual implementation is in Bandit._learn_one)."""
        self._learn_one(x, y)
        return self

    def add_models(self, new_models: typing.List[base.Estimator]):
        length_new_models = len(new_models)
        self.models += new_models
        self._n_arms += length_new_models
        self._N += [0] * length_new_models
        self.average_reward += [0.0] * length_new_models

    def _learn_one(self, x, y):
        """Updates the chosen model and the arm internals.

        It returns the reward value, the instantaneous metric value and the chosen_arm.

        """

        # Explore all arms pulled less than `explore_each_arm` times
        never_pulled_arm = [
            i for (i, n) in enumerate(self._N) if n < self.explore_each_arm
        ]
        chosen_arm = (
            self._rng.choice(never_pulled_arm) if never_pulled_arm else self._pull_arm()
        )

        # Predict and learn with the chosen model
        chosen_model = self[chosen_arm]
        y_pred = chosen_model.predict_one(x)
        self.metric.update(y_pred=y_pred, y_true=y)
        chosen_model.learn_one(x=x, y=y)

        # If warm up, do nothing (no updates of bandit internals).
        if self.warm_up and (self._n_iter == self.start_after):
            self._n_iter = (
                0  # must be reset to 0 since it is an input to some model (UCB)
            )
            self.warm_up = False

        self._n_iter += 1

        reward = self._compute_scaled_reward(y_pred=y_pred, y_true=y)

        # If not in warm up mode, update bandit internals for the pulled arm
        if not self.warm_up:
            self._update_bandit(chosen_arm=chosen_arm, reward=reward)

        return reward, self.metric._eval(y_pred, y), chosen_arm

    def _update_bandit(self, chosen_arm, reward):
        """Updates the bandit's arm internals."""
        # Updates common to all bandits
        self._N[chosen_arm] += 1
        self.average_reward[chosen_arm] += (1.0 / self._N[chosen_arm]) * (
            reward - self.average_reward[chosen_arm]
        )

        # Specific updates of the arm (for certain bandit model)
        self._update_arm(chosen_arm, reward)

    def _compute_scaled_reward(self, y_pred, y_true):
        """Compute the reward defined as the sigmoid of the metric chosen by the user.

        For regression, the target (y_true) and the prediction (y_pred) are standardized so that the reward is invariant to the scale of the problem.

        """

        # Scaling y so the reward distribution doesn't depend on the scale of y
        y_true = self._y_scaler.learn_one(dict(y=y_true)).transform_one(dict(y=y_true))[
            "y"
        ]
        y_pred = self._y_scaler.transform_one(dict(y=y_pred))["y"]

        metric_value = self.metric._eval(y_pred, y_true)
        metric_value = (
            metric_value if self.metric.bigger_is_better else (-1) * metric_value
        )
        reward = 2 * sigmoid(metric_value)  # multiply per 2 to have reward in [0, 1]

        return reward


class EpsilonGreedyBandit(Bandit):
    def __init__(
        self,
        models: typing.List[base.Estimator],
        metric: metrics.Metric,
        epsilon: float = 0.1,
        epsilon_decay: float = None,
        explore_each_arm: int = 1,
        start_after: int = 20,
        seed: int = None,
    ):
        super().__init__(
            models=models,
            metric=metric,
            explore_each_arm=explore_each_arm,
            start_after=start_after,
            seed=seed,
        )
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if epsilon_decay:
            self._starting_epsilon = epsilon

    def _pull_arm(self):
        chosen_arm = (
            argmax(self.average_reward, self._rng)
            if (self._rng.random() > self.epsilon)
            else self._rng.choice(range(self._n_arms))
        )

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The other arm internals are already updated in the method `Bandit._learn_one`.
        if self.epsilon_decay:
            self.epsilon = self._starting_epsilon * math.exp(
                -self._n_iter * self.epsilon_decay
            )


class EpsilonGreedyRegressor(EpsilonGreedyBandit, base.Regressor):
    r"""Epsilon-greedy bandit algorithm for regression.

    This bandit selects the best arm (defined as the one with the highest average reward) with
    probability $(1 - \epsilon)$ and draws a random arm with probability $\epsilon$. It is also
    called Follow-The-Leader (FTL) algorithm.

    Parameters
    ----------
    models
        The models to compare.
    metric
        Metric used for comparing models with.
    epsilon
        Exploration parameter (default : 0.1).
    epsilon_decay
        Exponential decay factor applied to epsilon.
    explore_each_arm
        The number of times each arm should explored first.
    start_after
        The number of iteration after which the bandit mechanism should begin.
    seed
        The seed for the algorithm (since not deterministic).

    Examples
    --------
    Let's use `UCBRegressor` to select the best learning rate for a linear regression model. First,
    we define the grid of models:

    >>> from river import compose
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import optim

    >>> models = [
    ...     compose.Pipeline(
    ...         preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     )
    ...     for lr in [1e-4, 1e-3, 1e-2, 1e-1]
    ... ]

    We decide to use TrumpApproval dataset:

    >>> from river import datasets
    >>> dataset = datasets.TrumpApproval()

    The chosen bandit is epsilon-greedy:

    >>> from river.expert import EpsilonGreedyRegressor
    >>> bandit = EpsilonGreedyRegressor(models=models, seed=1)

    The models in the bandit can then be trained in an online fashion.

    >>> for x, y in dataset:
    ...     bandit = bandit.learn_one(x=x, y=y)

    We can inspect the number of times (in percentage) each arm has been pulled.

    >>> for model, pct in zip(bandit.models, bandit.percentage_pulled):
    ...     lr = model["LinearRegression"].optimizer.learning_rate
    ...     print(f"{lr:.1e} — {pct:.2%}")
    1.0e-04 — 3.47%
    1.0e-03 — 2.85%
    1.0e-02 — 44.75%
    1.0e-01 — 48.93%

    The average reward of each model is also available:

    >>> for model, avg in zip(bandit.models, bandit.average_reward):
    ...     lr = model["LinearRegression"].optimizer.learning_rate
    ...     print(f"{lr:.1e} — {avg:.2f}")
    1.0e-04 — 0.00
    1.0e-03 — 0.00
    1.0e-02 — 0.56
    1.0e-01 — 0.01

    We can also select the best model (the one with the highest average reward).

    >>> best_model = bandit.best_model

    The learning rate chosen by the bandit is:

    >>> best_model["LinearRegression"].intercept_lr.learning_rate
    0.01

    References
    ----------
    [^1]: [Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.](http://incompleteideas.net/book/RLbook2020.pdf)
    [^2]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)
    [^3]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)

    """

    def __init__(
        self,
        models: typing.List[base.Estimator],
        metric: metrics.RegressionMetric = None,
        epsilon: float = 0.1,
        epsilon_decay: float = None,
        explore_each_arm: int = 3,
        start_after: int = 20,
        seed: int = None,
    ):

        if metric is None:
            metric = metrics.MAE()

        super().__init__(
            models=models,
            metric=metric,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            explore_each_arm=explore_each_arm,
            start_after=start_after,
            seed=seed,
        )

    @classmethod
    def _unit_test_params(cls):
        return {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01)),
                ),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.1)),
                ),
            ],
            "metric": metrics.MAE(),
        }

    def _pred_func(self, model):
        return model.predict_one


class UCBBandit(Bandit):
    def __init__(
        self,
        models: typing.List[base.Estimator],
        metric: metrics.Metric,
        delta: float = None,
        explore_each_arm: int = 1,
        start_after: int = 20,
        seed: int = None,
    ):

        if explore_each_arm < 1:
            raise ValueError("Argument 'explore_each_arm' should be >= 1")
        super().__init__(
            models=models,
            metric=metric,
            explore_each_arm=explore_each_arm,
            start_after=start_after,
            seed=seed,
        )
        if delta is not None and (delta >= 1 or delta <= 0):
            raise ValueError(
                "The parameter delta should be comprised in ]0, 1[ (or set to None)"
            )
        self.delta = delta

    def _pull_arm(self):
        if self.delta:
            exploration_bonus = [
                math.sqrt(2 * math.log(1 / self.delta) / n) for n in self._N
            ]
        else:
            exploration_bonus = [
                math.sqrt(2 * math.log(self._n_iter) / n) for n in self._N
            ]
        upper_bound = [
            avg_reward + exploration
            for (avg_reward, exploration) in zip(self.average_reward, exploration_bonus)
        ]
        chosen_arm = argmax(upper_bound, self._rng)

        return chosen_arm

    def _update_arm(self, arm, reward):
        # The arm internals are already updated in the method `Bandit._learn_one`.
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
    explore_each_arm
        The number of times each arm should explored first.
    start_after
        The number of iteration after which the bandit mechanism should begin.
    seed
        The seed for the algorithm (since not deterministic).

    Examples
    --------
    Let's use `UCBRegressor` to select the best learning rate for a linear regression model. First,
    we define the grid of models:

    >>> from river import compose
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> from river import optim

    >>> models = [
    ...     compose.Pipeline(
    ...         preprocessing.StandardScaler(),
    ...         linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     )
    ...     for lr in [1e-4, 1e-3, 1e-2, 1e-1]
    ... ]

    We decide to use TrumpApproval dataset:

    >>> from river import datasets
    >>> dataset = datasets.TrumpApproval()

    We use the UCB bandit:

    >>> from river.expert import UCBRegressor
    >>> bandit = UCBRegressor(models=models, seed=1)

    The models in the bandit can be trained in an online fashion.

    >>> for x, y in dataset:
    ...     bandit = bandit.learn_one(x=x, y=y)

    We can inspect the number of times (in percentage) each arm has been pulled.

    >>> for model, pct in zip(bandit.models, bandit.percentage_pulled):
    ...     lr = model["LinearRegression"].optimizer.learning_rate
    ...     print(f"{lr:.1e} — {pct:.2%}")
    1.0e-04 — 2.45%
    1.0e-03 — 2.45%
    1.0e-02 — 92.25%
    1.0e-01 — 2.85%

    The average reward of each model is also available:

    >>> for model, avg in zip(bandit.models, bandit.average_reward):
    ...     lr = model["LinearRegression"].optimizer.learning_rate
    ...     print(f"{lr:.1e} — {avg:.2f}")
    1.0e-04 — 0.00
    1.0e-03 — 0.00
    1.0e-02 — 0.74
    1.0e-01 — 0.05

    We can also select the best model (the one with the highest average reward).

    >>> best_model = bandit.best_model

    The learning rate chosen by the bandit is:

    >>> best_model["LinearRegression"].intercept_lr.learning_rate
    0.01

    References
    ----------
    [^1]: [Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3), 235-256.](https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf)
    [^2]: [Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.](https://tor-lattimore.com/downloads/book/book.pdf)
    [^3]: [Rivasplata, O. (2012). Subgaussian random variables: An expository note. Internet publication, PDF.]: (https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)

    """

    def __init__(
        self,
        models: typing.List[base.Estimator],
        metric: metrics.RegressionMetric = None,
        delta: float = None,
        explore_each_arm: int = 1,
        start_after: int = 20,
        seed: int = None,
    ):

        if metric is None:
            metric = metrics.MAE()

        super().__init__(
            models=models,
            metric=metric,
            delta=delta,
            explore_each_arm=explore_each_arm,
            start_after=start_after,
            seed=seed,
        )

    @classmethod
    def _unit_test_params(cls):
        return {
            "models": [
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.01)),
                ),
                compose.Pipeline(
                    preprocessing.StandardScaler(),
                    linear_model.LinearRegression(optimizer=optim.SGD(lr=0.1)),
                ),
            ],
            "metric": metrics.MAE(),
        }

    def _pred_func(self, model):
        return model.predict_one
