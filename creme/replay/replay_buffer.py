from . import base


class ReplayBufferRegressor(base.ReplayBuffer, base.BufferRegressor):
    """ReplayBufferRegressor

    Re-train the model on the observations where it produces important errors.

    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data.

    New observations will be stored if in the buffer if loss is sufficiently high compared to the
    losses already stored in the buffer.

    Parameters:
        regressor (base.Regressor): Selected regressor.
        loss_function (creme.optim.losses): Criterion to store observations in the buffer.
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.

    """
    def __init__(self, regressor, loss_function, size, p, seed = None):
        base.ReplayBuffer.__init__(self, p=p, size=size, seed=seed)
        base.BufferRegressor.__init__(self, regressor=regressor, seed=seed)
        self.loss_function = loss_function

    def fit_one(self, x, y):
        """
        Update the model with observations stored in the buffer with a probability p. Update the
        model with input observation with a probability (1 - p).

        Parameters:
            x (Dict): Features.
            y (float): Target.

        """
        # Eval loss of the model when predicting the input observation.
        loss = self.loss_function.eval(y_true=y, y_pred=self.predict_one(x))

        # If the loss is high enough, the tuple (x, y) will be stored in the buffer.
        self._update_buffer(x, y, loss)

        # Replace the input (x, y) with an observation from the buffer with a probability p.
        key, x, y = self._training_sample_selection(x, y)

        # Update the model.
        self.regressor.fit_one(x, y)

        if key is not None:
            # If the key is defined, it means that whe have replaced the input (x, y) with an
            # observation from the buffer. After fitting this bufferized observation, we update it's
            # loss stored in the buffer.
            loss = self.loss_function.eval(y_true=y, y_pred=self.predict_one(x))
            self._update_criterion(key, loss)

        return self


class ReplayBufferClassifier(base.ReplayBuffer, base.BufferClassifier):
    """ReplayBufferClassifier

    Re-train the model on the observations where it produces important errors.

    ReplayBuffer stores a dictionnary of tuples  (features, target). The tuples stored in the
    buffer are the observations which are difficult for a model to regress. When fitting the model,
    it is re-trained with probability p on one of the observations (sampled uniformly) in the
    buffer. The model is re-trained with a probability (1 - p) on the new data.

    New observations will be stored if in the buffer if loss is sufficiently high compared to the
    losses already stored in the buffer.

    Parameters:
        classifier (base.Classifier): Selected classifier.
        loss_function (creme.optim.losses): Criterion to store observations in the buffer.
        size (int): Number of stored tuples (features, target).
        p (float): Probability to update the model with an observation from the buffer when fitting
            on a new observation. 0. <= p <= 1.
        seed (int): Random state.

    """
    def __init__(self, classifier, loss_function, size, p, seed = None):
        base.ReplayBuffer.__init__(self, p=p, size=size, seed=seed)
        base.BufferClassifier.__init__(self, classifier=classifier, seed=seed)
        self.loss_function = loss_function

    def fit_one(self, x, y):
        # Eval loss of the model when predicting the input observation.
        loss = self.loss_function.eval(y_true=y, y_pred=self.predict_proba_one(x))

        # If the loss is high enough, the tuple (x, y) will be stored in the buffer.
        self._update_buffer(x, y, loss)

        # Replace the input (x, y) with an observation from the buffer with a probability p.
        key, x, y = self._training_sample_selection(x, y)

        # Update the model.
        self.classifier.fit_one(x, y)

        if key is not None:
            # If the key is defined, it means that whe have replaced the input (x, y) with an
            # observation from the buffer. After fitting this bufferized observation, we update it's
            # loss stored in the buffer.
            loss = self.loss_function.eval(y_true=y, y_pred=self.predict_proba_one(x))
            self._update_criterion(key, loss)

        return self
