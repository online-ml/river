import abc
import math

from scipy import special

from .. import base


class BaseNB(base.MultiClassifier, abc.ABC):
    """Base Naive Bayes class.

    This class inherits ``predict_one`` from ``base.MultiClassifier``.

    """

    @abc.abstractmethod
    def joint_log_likelihood(self, x):
        """Compute the unnormalized posterior log-likelihood of x.

        The log-likelihood is ``log P(c) + log P(x|c)``

        """

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self.joint_log_likelihood(x)
        if not jll:
            return {}
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}
