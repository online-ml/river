from . import base


class ReplayBufferRegressor(base.ReplayBuffer):
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

    """
    def __init__(self, regressor, loss_function, size, p, seed=None):
        base.ReplayBuffer.__init__(self, model=regressor, pred_func=regressor.predict_one,
            loss_function=loss_function, size=size, p=p, seed=seed
        )

    @property
    def _model(self):
        return self.model

    def predict_one(self, x):
        return self.model.predict_one(x)


class ReplayBufferClassifier(base.ReplayBuffer):
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

    """
    def __init__(self, classifier, loss_function, size, p, seed=None):
        base.ReplayBuffer.__init__(self, model=classifier, pred_func=classifier.predict_proba_one,
            loss_function=loss_function, size=size, p=p, seed=seed
        )

    @property
    def _model(self):
        return self.model

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def predict_one(self, x):
        return self.model.predict_one(x)
