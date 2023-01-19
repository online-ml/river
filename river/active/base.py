import random
from river import base


class ClassificationSampler(base.Wrapper, base.Classifier):
    """Base class for active learning classifiers.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    seed
        Random number generator seed for reproducibility.

    """

    def __init__(self, classifier: base.Classifier, seed: int = None):
        self.classifier = classifier
        self.seed = seed
        self_rng = random.Random(seed)

    @property
    def _wrapped_model(self):
        return self.classifier

    def learn_one(self, x, y):

        # Update the quantiles
        error = y - self.regressor.predict_one(x)
        self._lower.update(error)
        self._upper.update(error)

        self.regressor.learn_one(x, y)

        return self
