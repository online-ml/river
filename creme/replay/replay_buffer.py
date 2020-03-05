import random

import collections

from .. import base
from .. import utils


class Triplet(collections.namedtuple('Triplet', 'x y loss')):

    def __lt__(self, other):

        return self.loss < other.loss



class ReplayBuffer(base.Wrapper):
    """ReplayBuffer

    Re-train the model on the observations where it produces important errors.
    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data. When the model fit
    on an observation from the buffer, the loss is re-computed to refresh the buffer. New
    observations will be stored if in the buffer if loss is sufficiently high compared to the losses
    already stored in the buffer.

    Parameters:
        model (base.Estimator): Selected model.
        loss_function (creme.optim.losses): Criterion to store observations in the buffer.
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.
    """
    def __init__(self, model, loss_function, size, p,seed = None):
        self.model = model
        self.loss_function = loss_function
        self.pred_func = model.predict_one

        if isinstance(model, base.Classifier):
            self.pred_func = model.predict_proba_one

        self.p = p
        self.size = size
        self.buffer = utils.SortedWindow(self.size)
        self.seed = seed
        self._rng = random.Random(seed)

    def fit_one(self, x, y):
        """
        Compute the loss of the input observation. If the loss is high enough,  the tuple (x, y)
        will be stored in the buffer. Update the model with observations stored in the buffer with a
        probability p. Update the model with input observation with a probability (1 - p). Compute
        the loss a second times if the model fitted an observation from the buffer.

        Parameters:
            x (Dict): Features.
            y (float): Target.

        """
        # Eval loss of the model when predicting the input observation.
        loss = self.loss_function.eval(y_true=y, y_pred=self.pred_func(x))

        # If the buffer is not full:
        if len(self.buffer) < self.size:
            self.buffer.append(Triplet(x=x, y=y, loss=loss))

        # If the buffer is fulle and the input loss is higher than the minimal loss storred.
        elif loss > self.buffer[0].loss:
            self.buffer.pop(0)
            self.buffer.append(Triplet(x=x, y=y, loss=loss))

        if self._rng.uniform(0, 1) <= self.p:
            i = self._rng.randint(0, len(self.buffer) - 1)

            triplet = self.buffer.pop(i)

            self.model.fit_one(triplet.x, triplet.y)

            # After fitting this bufferized observation, we update it's loss stored in the buffer.
            loss = self.loss_function.eval(y_true=triplet.y, y_pred=self.pred_func(triplet.x))

            self.buffer.append(Triplet(x=triplet.x, y=triplet.y, loss=loss))

        # Probability (1 - p).
        else:
            self.model.fit_one(x, y)

        return self


class ReplayBufferRegressor(ReplayBuffer):
    """ReplayBufferRegressor

    Re-train the model on the observations where it produces important errors.
    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data. When the model fit
    on an observation from the buffer, the loss is re-computed to refresh the buffer. New
    observations will be stored if in the buffer if loss is sufficiently high compared to the losses
    already stored in the buffer.

    Parameters:
        Regressor (base.Regressor): Selected model.
        loss_function (creme.optim.losses): Criterion to store observations in the buffer.
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import replay

            >>> model = preprocessing.StandardScaler()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     replay.ReplayBufferRegressor(
            ...         regressor = linear_model.LinearRegression(),
            ...         loss_function = optim.losses.Absolute(),
            ...          p = 0.2,
            ...          size = 30,
            ...         seed = 42,
            ...     )
            ... )

            >>> model_selection.progressive_val_score(
            ...     datasets.TrumpApproval(),
            ...     model,
            ...     metrics.MAE(),
            ...     print_every = 500
            ... )
            [500] MAE: 3.240558
            [1,000] MAE: 1.949106
            MAE: 1.947374

    """
    def __init__(self, regressor, loss_function, size, p, seed=None):
        ReplayBuffer.__init__(self, model=regressor, loss_function=loss_function, size=size,
            p=p, seed=seed
        )

    @property
    def _model(self):
        return self.model

    def predict_one(self, x):
        return self.model.predict_one(x)


class ReplayBufferClassifier(ReplayBuffer):
    """ReplayBufferClassifier

    Re-train the model on the observations where it produces important errors.
    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data. When the model fit
    on an observation from the buffer, the loss is re-computed to refresh the buffer. New
    observations will be stored if in the buffer if loss is sufficiently high compared to the losses
    already stored in the buffer.

    Parameters:
        classifier (base.Classifier): Selected model.
        loss_function (creme.optim.losses): Criterion to store observations in the buffer.
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.


    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import replay

            >>> model = preprocessing.StandardScaler()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     replay.ReplayBufferClassifier(
            ...         classifier = linear_model.LogisticRegression(),
            ...         loss_function = optim.losses.CrossEntropy(),
            ...          p = 0.1,
            ...          size = 30,
            ...         seed = 42,
            ...     )
            ... )

            >>> model_selection.progressive_val_score(
            ...     datasets.CreditCard(),
            ...     model,
            ...     metrics.ROCAUC(),
            ...     print_every = 45_000
            ... )
            [45,000] ROCAUC: 0.927917
            [90,000] ROCAUC: 0.924025
            [135,000] ROCAUC: 0.920276
            [180,000] ROCAUC: 0.916839
            [225,000] ROCAUC: 0.91115
            [270,000] ROCAUC: 0.914077
            ROCAUC: 0.911448

    """
    def __init__(self, classifier, loss_function, size, p, seed=None):
        ReplayBuffer.__init__(self, model=classifier, loss_function=loss_function, size=size,
            p=p, seed=seed
        )

    @property
    def _model(self):
        return self.model

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def predict_one(self, x):
        return self.model.predict_one(x)
