from . import base


class EntropySampler(base.ClassificationSampler):
    """Entropy active learning sampler.

    Parameters
    ----------
    classifier
        The classifier to wrap.
    seed
        Random number generator seed for reproducibility.

    """
