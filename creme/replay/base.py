import numpy as np

import collections
import random

from .. import base


class ReplayBuffer:
    """ReplayBuffer

    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress or classify.The buffer is
    used to re-train a model on targets that are difficult to extrapolate. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data.

    To update the buffer, you must specify the loss of a given tuple (features, target). This new
    observation will be stored if the loss is sufficiently high compared to the losses already
    stored in the buffer.

    Parameters:
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.

    """
    def __init__(self, size, p, seed=None):
        self.size = size
        self.p = p
        self.buffer = {}
        # Intiliaze the losses history with respect to the buffer size with losses equals to -1.
        self.loss_history = {key: -1. for key in range(self.size)}
        # Initialize the minimum loss observed to -1.
        self.min_loss = -1.
        # Initialize the key associated to the minimum loss observed.
        self.key_min_loss = 0
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def _update_buffer(self, x, y, loss):
        """
        If the loss of the new observation is greater than the smallest loss stored in the buffer,
        the new observation (x, y) will replace the observation associated with the lowest loss.

        Parameters:
            x (dict): Features.
            y (float): Target.
            loss (float): Error of the model.
        """
        if loss > self.min_loss:
            self.buffer[self.key_min_loss] = (x, y)
            self.loss_history[self.key_min_loss] = loss

            # This operation as a high cost when buffer is large.
            # To reduce this cost we look for the minimum stored loss only when updating the buffer.
            self.key_min_loss = min(self.loss_history.keys(), key=(lambda k: self.loss_history[k]))
            self.min_loss = self.loss_history[self.key_min_loss]
        return self

    def _update_criterion(self, key, loss):
        """
        _update_criterion is called when the model fit on a observation (x, y) already stored in
        the buffer. _update_criterion aim at re-computing the loss associated to the key of the
        stored observation (x, y).

        Parameters:
            key (int): Key of the input observation stored in the buffer
            loss (float): Error of the model.
        """
        self.loss_history[key] = loss
        if loss < self.min_loss:
            self.min_loss = loss
            self.key_min_loss = key
        return self

    def _training_sample_selection(self, x, y):
        """
        Randomly returns a tuple (features, target) and its membership key from the buffer with a
        probability p. The key is used to recalculate the loss stored in loss_history when updating
        the model on a tuple of the buffer. Returns the input tuple (features, target) with a
        probability (1 - p).

        Parameters:
            x (dict): Features.
            y (float): Target.
            loss (float): Error of the model.

        Returns:
            key (Union[Int, float_]): The key is only defined when a buffer observation has been
                selected.
            x (dict): Set of features.
            y (float): Target.
        """
        if self._rng.uniform(0, 1) <= self.p:
            # Probability p:
            key = self._rng.choice(list(self.buffer.keys()))
            x, y = self.buffer[key]
            return key, x, y
        else:
            # Probability (1 - p):
            return None, x, y



class BufferRegressor(base.Wrapper, base.Regressor):

    def __init__(self, regressor, seed=None):
        self.regressor = regressor
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _model(self):
        return self.regressor

    def predict_one(self, x):
        return self.regressor.predict_one(x)


class BufferClassifier(base.Wrapper, base.Classifier):

    def __init__(self, classifier, seed=None):
        self.classifier = classifier
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _model(self):
        return self.classifier

    def predict_proba_one(self, x):
        return self.classifier.predict_proba_one(x)

    def predict_one(self, x):
        return self.classifier.predict_one(x)
