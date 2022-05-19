from river import metrics
from river._bandit import UCB
from river.model_selection.base import BanditRegressor


class UCBRegressor(BanditRegressor):
    """Model selection based on the UCB bandit strategy.

    Due to the nature of this algorithm, it's recommended to scale the target so that it exhibits
    sub-gaussian properties. This can be done by using a `preprocessing.TargetStandardScaler`.

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
    ...     for lr in [0.0001, 0.001, 1e-05, 0.01]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     preprocessing.TargetStandardScaler(
    ...         model_selection.UCBRegressor(
    ...             models,
    ...             delta=1,
    ...             burn_in=0,
    ...             seed=42
    ...         )
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.410815

    >>> model['TargetStandardScaler'].regressor.bandit
    Ranking   MAE        Pulls   Share
         #3   1.441458       8    0.80%
         #1   0.291200     242   24.18%
         #2   0.808878      19    1.90%
         #0   0.204892     732   73.13%

    >>> model['TargetStandardScaler'].regressor.best_model
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
    [^1]: [Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1), 4-22.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.674.1620&rep=rep1&type=pdf)
    [^2]: [Upper Confidence Bounds - The Multi-Armed Bandit Problem and Its Solutions - Lilian Weng](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#upper-confidence-bounds)
    [^3]: [The Upper Confidence Bound Algorithm - Bandit Algorithms](https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)

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
