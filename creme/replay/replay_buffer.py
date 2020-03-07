import random

import collections

from .. import base
from .. import utils


class Triplet(collections.namedtuple('Triplet', 'x y loss')):

    def __lt__(self, other):
        return self.loss < other.loss


class ReplayBuffer(base.Wrapper):
    """ReplayBuffer

    Stores the hardest data to fit in a buffer. When the fit_one method is called, the model
    is updated on an observation of the buffer with a probability p or updated with the current
    observation with a probability (1 - p).

    The model systematically evaluates the difficulty of an input data of the fit_one method and
    stores it in the buffer if the associated loss to this observation is greater than the smallest
    loss already storred in the buffer.

    If a new observation is storred and if the buffer is full, then the new observation will take
    the place of the observation associated with the smallest loss.

    If the buffer is not full, the input observation is systematically added to the buffer.

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

        elif isinstance(model, base.BinaryClassifier):
            self.pred_func = lambda x: model.predict_proba_one(x)[True]

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

    Stores the hardest data to fit in a buffer. When the fit_one method is called, the model
    is updated on an observation of the buffer with a probability p or updated with the current
    observation with a probability (1 - p).

    The model systematically evaluates the difficulty of an input data of the fit_one method and
    stores it in the buffer if the associated loss to this observation is greater than the smallest
    loss already storred in the buffer.

    If a new observation is storred and if the buffer is full, then the new observation will take
    the place of the observation associated with the smallest loss.

    If the buffer is not full, the input observation is systematically added to the buffer.

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

    Stores the hardest data to fit in a buffer. When the fit_one method is called, the model
    is updated on an observation of the buffer with a probability p or updated with the current
    observation with a probability (1 - p).

    The model systematically evaluates the difficulty of an input data of the fit_one method and
    stores it in the buffer if the associated loss to this observation is greater than the smallest
    loss already storred in the buffer.

    If a new observation is storred and if the buffer is full, then the new observation will take
    the place of the observation associated with the smallest loss.

    If the buffer is not full, the input observation is systematically added to the buffer.

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
            ...          size = 40,
            ...         seed = 42,
            ...     )
            ... )

            >>> model_selection.progressive_val_score(
            ...     X_y = datasets.Phishing(),
            ...     model = model,
            ...     metric = metrics.ROCAUC(),
            ...     print_every = 500,
            ... )
            [500] ROCAUC: 0.928271
            [1,000] ROCAUC: 0.948547
            ROCAUC: 0.952755

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
