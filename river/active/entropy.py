from . import base


class EntropySampler(base.ActiveLearningClassifier):
    """Entropy active learning sampler.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    seed
        Random number generator seed for reproducibility.

    """

    def _ask_for_label(self, x, y_pred) -> bool:
        return self._rng.random() < 0.5
