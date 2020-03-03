import numpy as np

import random

from .. import base


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
    def __init__(self, model, pred_func, loss_function, size, p,seed = None):

        self.model = model
        self.pred_func = pred_func
        self.loss_function = loss_function

        self.size = size
        self.p = p

        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.buffer = {}
        self.loss_history = {key: -1. for key in range(self.size)}

        self.min_loss = -1.
        self.key_min_loss = 0

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

        # If the loss is high enough, the observation (x, y) will replace observation associated to
        # the lowest loss value stored in the buffer.
        if loss > self.min_loss:
            # Store the new observation in the buffer.
            self.buffer[self.key_min_loss] = (x, y)
            # Store the new loss.
            self.loss_history[self.key_min_loss] = loss

            # Update the new minimum loss and get the key of the observation corresponding to the
            # minimum loss.
            self.key_min_loss = min(self.loss_history.keys(), key=(lambda k: self.loss_history[k]))
            self.min_loss = self.loss_history[self.key_min_loss]

        # Enter in the condition with a probability p.
        if self._rng.uniform(0, 1) <= self.p:
            key = self._rng.choice(list(self.buffer.keys()))
            x, y = self.buffer[key]

            # Update the model with the selected observation.
            self.model.fit_one(x, y)

            # If the key is defined, it means that whe have replaced the input (x, y) with an
            # observation from the buffer. After fitting this bufferized observation, we update it's
            # loss stored in the buffer.
            loss = self.loss_function.eval(y_true=y, y_pred=self.pred_func(x))
            self.loss_history[key] = loss

            if loss < self.min_loss:
                self.min_loss = loss
                self.key_min_loss = key

        # Probability (1 - p).
        else:
            self.model.fit_one(x, y)

        return self
