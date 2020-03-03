import numpy as np

from .. import base


class BufferRegressor(base.Wrapper, base.Regressor):

    def __init__(self, model, seed=None):
        self.model = model
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _model(self):
        return self.model

    def predict_one(self, x):
        return self.model.predict_one(x)


class BufferClassifier(base.Wrapper, base.Classifier):

    def __init__(self, model, seed=None):
        self.model = model
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def _model(self):
        return self.model

    def predict_proba_one(self, x):
        return self.model.predict_proba_one(x)

    def predict_one(self, x):
        return self.model.predict_one(x)
