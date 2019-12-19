import numpy as np

from .. import base


class Sampler(base.Wrapper, base.Classifier):

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
