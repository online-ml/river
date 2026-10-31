from __future__ import annotations

import collections
import collections.abc
import random
import typing

from river import base, linear_model, optim, utils

T = typing.TypeVar("T", base.Classifier, base.Regressor)


class Triplet(collections.namedtuple("Triplet", "x y loss")):
    def __lt__(self, other: typing.Any) -> bool:
        return typing.cast(float, self.loss) < typing.cast(float, other.loss)


class HardSampling(base.Wrapper[T], typing.Generic[T]):
    """Hard sampler."""

    def __init__(
        self,
        model: T,
        loss: optim.base.Loss,
        size: int,
        p: float,
        seed: int | None = None,
    ):
        self.model: T = model
        self.loss = loss
        self.p = p
        self.size = size
        self.buffer: utils.SortedWindow[Triplet] = utils.SortedWindow(self.size)
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def _wrapped_model(self) -> T:
        return self.model

    @property
    def _model_pred_func(
        self,
    ) -> collections.abc.Callable[[dict[base.typing.FeatureName, typing.Any]], typing.Any]:
        terminal = base._unwrap(self.model)
        if isinstance(terminal, base.Classifier):
            model = typing.cast(base.typing.ProbabilisticPredictor, self.model)
            if not terminal._multiclass:
                return lambda x: model.predict_proba_one(x)[True]
            return model.predict_proba_one
        return typing.cast(
            collections.abc.Callable[[dict[base.typing.FeatureName, typing.Any]], typing.Any],
            self.model.predict_one,
        )

    def learn_one(
        self,
        x: dict[base.typing.FeatureName, typing.Any],
        y: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        loss = typing.cast(float, self.loss(y_true=y, y_pred=self._model_pred_func(x)))

        # Copy x so the caller can safely mutate the input after learn_one
        # without disturbing the buffered triplets.
        if len(self.buffer) < self.size:
            self.buffer.append(Triplet(x=dict(x), y=y, loss=loss))

        elif loss > self.buffer[0].loss:
            self.buffer.pop(0)

            self.buffer.append(Triplet(x=dict(x), y=y, loss=loss))

        # Probability p
        if self._rng.uniform(0, 1) <= self.p:
            i = self._rng.randint(0, len(self.buffer) - 1)

            triplet = self.buffer.pop(i)

            self.model.learn_one(triplet.x, triplet.y, **kwargs)

            loss = typing.cast(
                float, self.loss(y_true=triplet.y, y_pred=self._model_pred_func(triplet.x))
            )

            self.buffer.append(Triplet(x=triplet.x, y=triplet.y, loss=loss))

        # Probability (1 - p)
        else:
            self.model.learn_one(x, y)


class HardSamplingRegressor(HardSampling[base.Regressor], base.Regressor):
    """Hard sampling regressor.

    This wrapper enables a model to retrain on past samples who's output was hard to predict.
    This works by storing the hardest samples in a buffer of a fixed size. When a new sample
    arrives, the wrapped model is either trained on one of the buffered samples with a probability
    p or on the new sample with a probability (1 - p).

    The hardness of an observation is evaluated with a loss function that compares the sample's
    ground truth with the wrapped model's prediction. If the buffer is not full, then the sample
    is added to the buffer. If the buffer is full and the new sample has a bigger loss than the
    lowest loss in the buffer, then the sample takes its place.

    Parameters
    ----------
    regressor
    size
        Size of the buffer.
    p
        Probability of updating the model with a sample from the buffer instead of a new incoming
        sample.
    loss
        Criterion used to evaluate the hardness of a sample.
    seed
        Random seed.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     imblearn.HardSamplingRegressor(
    ...         regressor=linear_model.LinearRegression(),
    ...         p=.2,
    ...         size=30,
    ...         seed=42,
    ...     )
    ... )

    >>> evaluate.progressive_val_score(
    ...     datasets.TrumpApproval(),
    ...     model,
    ...     metrics.MAE(),
    ...     print_every=500
    ... )
    [500] MAE: 2.274021
    [1,000] MAE: 1.392399
    [1,001] MAE: 1.391246
    MAE: 1.391246

    """

    def __init__(
        self,
        regressor: base.Regressor,
        size: int,
        p: float,
        loss: optim.losses.RegressionLoss | None = None,
        seed: int | None = None,
    ):
        if loss is None:
            loss = optim.losses.Absolute()
        super().__init__(model=regressor, loss=loss, size=size, p=p, seed=seed)

    @property
    def regressor(self) -> base.Regressor:
        return self.model

    def predict_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> base.typing.RegTarget:
        return self.model.predict_one(x, **kwargs)

    @classmethod
    def _unit_test_params(cls) -> collections.abc.Iterator[dict[str, typing.Any]]:
        yield {"regressor": linear_model.LinearRegression(), "p": 0.1, "size": 40}


class HardSamplingClassifier(HardSampling[base.Classifier], base.Classifier):
    """Hard sampling classifier.

    This wrapper enables a model to retrain on past samples who's output was hard to predict.
    This works by storing the hardest samples in a buffer of a fixed size. When a new sample
    arrives, the wrapped model is either trained on one of the buffered samples with a probability
    p or on the new sample with a probability (1 - p).

    The hardness of an observation is evaluated with a loss function that compares the sample's
    ground truth with the wrapped model's prediction. If the buffer is not full, then the sample
    is added to the buffer. If the buffer is full and the new sample has a bigger loss than the
    lowest loss in the buffer, then the sample takes its place.

    Parameters
    ----------
    classifier
    size
        Size of the buffer.
    p
        Probability of updating the model with a sample from the buffer instead of a new
        incoming sample.
    loss
        Criterion used to evaluate the hardness of a sample.
    seed
        Random seed.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import imblearn
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import optim
    >>> from river import preprocessing

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     imblearn.HardSamplingClassifier(
    ...         classifier=linear_model.LogisticRegression(),
    ...         p=0.1,
    ...         size=40,
    ...         seed=42,
    ...     )
    ... )

    >>> evaluate.progressive_val_score(
    ...     dataset=datasets.Phishing(),
    ...     model=model,
    ...     metric=metrics.ROCAUC(),
    ...     print_every=500,
    ... )
    [500] ROCAUC: 92.78%
    [1,000] ROCAUC: 94.76%
    [1,250] ROCAUC: 95.06%
    ROCAUC: 95.06%

    """

    def __init__(
        self,
        classifier: base.Classifier,
        size: int,
        p: float,
        loss: optim.losses.BinaryLoss | optim.losses.MultiClassLoss | None = None,
        seed: int | None = None,
    ):
        if loss is None:
            loss = optim.losses.CrossEntropy() if classifier._multiclass else optim.losses.Log()
        super().__init__(model=classifier, loss=loss, size=size, p=p, seed=seed)

    @property
    def classifier(self) -> base.Classifier:
        return self.model

    @property
    def _multiclass(self) -> bool:
        return self.model._multiclass

    def predict_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> base.typing.ClfTarget | None:
        return self.model.predict_one(x, **kwargs)

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> dict[base.typing.ClfTarget, float]:
        return self.model.predict_proba_one(x, **kwargs)

    @classmethod
    def _unit_test_params(cls) -> collections.abc.Iterator[dict[str, typing.Any]]:
        yield {"classifier": linear_model.LogisticRegression(), "p": 0.1, "size": 40}
