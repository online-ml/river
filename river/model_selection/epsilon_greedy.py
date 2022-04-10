from river import metrics
from river._bandit import EpsilonGreedy

from .base import BanditRegressor


class EpsilonGreedyRegressor(BanditRegressor):
    r"""Model selection based on the $\eps$-greedy bandit strategy.

    Performs model selection by using an $\eps$-greedy bandit strategy. A model is selected for
    each learning step. The best model is selected (1 - $\eps$%) of the time.

    Selection bias is a common problem when using bandits for online model selection. This bias can
    be mitigated by using a burn-in phase. Each model is given the chance to learn during the first
    `burn_in` steps.

    Parameters
    ----------
    models
        The models to choose from.
    metric
        The metric that is used to compare models with each other. Defaults to `metrics.MAE`.
    epsilon
        The fraction of time exploration is performed rather than exploitation.
    decay
        Exponential factor at which `epsilon` decays.
    burn_in
        The number of initial steps during which each model is updated.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [0.0001, 0.001, 1e-05, 0.01]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.EpsilonGreedyRegressor(
    ...         models,
    ...         epsilon=0.1,
    ...         decay=0.001,
    ...         burn_in=100,
    ...         seed=1
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.363516

    >>> model['EpsilonGreedyRegressor'].bandit
    Ranking   MAE         Pulls   Share
         #2   15.850129     111    8.53%
         #1   13.060601     117    8.99%
         #3   16.519079     109    8.38%
         #0    1.387839     964   74.10%

    >>> model['EpsilonGreedyRegressor'].best_model
    LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
        )
      )
      loss=Squared ()
      l2=0.
      l1=0.
      intercept_init=0.
      intercept_lr=Constant (
        learning_rate=0.01
      )
      clip_gradient=1e+12
      initializer=Zeros ()
    )

    References
    ----------
    [^1]: [ε-Greedy Algorithm - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#%CE%B5-greedy-algorithm)

    """

    def __init__(
        self,
        models,
        metric=None,
        epsilon=0.1,
        decay=0.0,
        burn_in=100,
        seed: int = None,
    ):
        if metric is None:
            metric = metrics.MAE()
        super().__init__(
            models=models,
            metric=metric,
            policy=EpsilonGreedy(
                epsilon=epsilon,
                decay=decay,
                burn_in=burn_in,
                seed=seed,
            ),
        )

    @property
    def epsilon(self):
        return self.policy.epsilon

    @property
    def decay(self):
        return self.policy.decay

    @property
    def burn_in(self):
        return self.policy.burn_in

    @property
    def seed(self):
        return self.policy.seed
