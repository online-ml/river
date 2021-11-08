import typing

from river import optim
from river.linear_model.glm import GLM

from .base import AnomalyDetector


class OneClassSVM(GLM, AnomalyDetector):
    """One-class SVM for anomaly detection.

    It is encouraged to scale the data upstream with `preprocessing.StandardScaler`, as well as use
    `feature_extraction.RBFSampler` to capture non-linearities.

    Parameters
    ----------
    nu
        An upper bound on the fraction of training errors and a lower bound of the fraction of
        support vectors. You can think of it as the expected fraction of anomalies.
    optimizer
        The sequential optimizer used for updating the weights.
    intercept_lr
        Learning rate scheduler used for updating the intercept. A `optim.schedulers.Constant` is
        used if a `float` is provided. The intercept is not updated when this is set to 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Examples
    --------

    >>> from random import Random
    >>> from river import anomaly
    >>> from river import compose
    >>> from river import datasets
    >>> from river import feature_extraction
    >>> from river import optim
    >>> from river import preprocessing

    >>> def data():
    ...     rng = Random(42)
    ...     for _ in range(1000):
    ...         a = rng.uniform(0, 1)
    ...         b = rng.uniform(0, 1)
    ...         if rng.random() < 0.005:
    ...             a *= 10
    ...             b *= 10
    ...         yield {'a': a, 'b': b}


    >>> model = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     anomaly.QuantileThresholder(
    ...         anomaly.OneClassSVM(nu=0.2),
    ...         q=0.995
    ...     )
    ... )

    >>> for x in data():
    ...     score = model.score_one(x)
    ...     model = model.learn_one(x)
    ...     if score:
    ...         print(" ".join(f"{xi:.3f}" for xi in x.values()))
    0.545 0.220
    0.809 0.006
    0.698 0.340
    0.957 0.337
    0.097 0.847
    0.807 0.730
    4.236 2.767
    2.547 7.088
    9.503 3.273
    8.577 4.413
    2.188 5.264

    """

    def __init__(
        self,
        nu=0.1,
        optimizer: optim.Optimizer = None,
        intercept_lr: typing.Union[optim.schedulers.Scheduler, float] = 0.01,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer = None,
    ):
        super().__init__(
            optimizer=optim.SGD(0.01) if optimizer is None else optimizer,
            loss=optim.losses.Hinge(),
            intercept_init=1.0,
            intercept_lr=intercept_lr,
            l2=nu / 2,
            clip_gradient=clip_gradient,
            initializer=initializer if initializer else optim.initializers.Zeros(),
        )
        self.nu = nu

    def _get_intercept_update(self, loss_gradient):
        return (
            super()._get_intercept_update(loss_gradient)
            + 2.0 * self.intercept_lr.get(self.optimizer.n_iterations) * self.l2
        )

    def learn_one(self, x):
        return super().learn_one(x, y=1)

    def learn_many(self, X):
        return super().learn_many(x, y=pd.Series(True, index=X.index))

    def score_one(self, x):
        return self._raw_dot_one(x) - self.intercept
