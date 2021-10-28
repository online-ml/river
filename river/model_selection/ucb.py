import math
import operator

from river import metrics

from .bandit import BanditPolicy, BanditRegressor


class UCB(BanditPolicy):
    def __init__(self, delta, burn_in, seed):
        super().__init__(burn_in, seed)
        self.delta = delta

    def _pull(self, bandit):
        sign = operator.pos if bandit.metric.bigger_is_better else operator.neg
        upper_bounds = [
            sign(arm.metric.get())
            + self.delta * math.sqrt(2 * math.log(bandit.n_pulls) / arm.n_pulls)
            if arm.n_pulls
            else math.inf
            for arm in bandit.arms
        ]
        yield max(bandit.arms, key=lambda arm: upper_bounds[arm.index])


class UCBRegressor(BanditRegressor):
    """Model selection based on the UCB bandit strategy.

    Parameters
    ----------
    models
        The models to choose from.
    metric
        The metric that is used to compare models with each other. Defaults to `metrics.MAE`.
    delta
        Exploration parameter.
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
    ...     for lr in [1e-5, 1e-4, 1e-3, 1e-2]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.UCBRegressor(
    ...         models,
    ...         delta=50,
    ...         burn_in=0,
    ...         seed=42
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.9054

    >>> model['UCBRegressor'].bandit
    Ranking   MAE         Pulls   Share
          3   32.092731      26    2.60%
          2   31.911083      26    2.60%
          1   27.594521      33    3.30%
          0    1.332678     916   91.51%

    >>> model['UCBRegressor'].best_model
    LinearRegression (
      optimizer=SGD (
        lr=Constant (
          learning_rate=0.01
        )
      )
      loss=Squared ()
      l2=0.
      intercept_init=0.
      intercept_lr=Constant (
        learning_rate=0.01
      )
      clip_gradient=1e+12
      initializer=Zeros ()
    )

    References
    ----------
    [^1]: [Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1), 4-22.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.674.1620&rep=rep1&type=pdf)

    """

    def __init__(
        self,
        models,
        metric=None,
        delta=1,
        burn_in=100,
        seed: int = None,
    ):
        if metric is None:
            metric = metrics.MAE()
        super().__init__(
            models=models,
            metric=metric,
            policy=UCB(
                delta=delta,
                burn_in=burn_in,
                seed=seed,
            ),
        )

    @property
    def delta(self):
        return self.policy.delta

    @property
    def burn_in(self):
        return self.policy.burn_in

    @property
    def seed(self):
        return self.policy.seed
