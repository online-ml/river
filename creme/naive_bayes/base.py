import abc
import math

from scipy import special

from .. import base


class BaseNB(base.MultiClassifier, abc.ABC):
    """Base Naive Bayes class.

    This class inherits ``predict_one`` from ``base.MultiClassifier``.

    """

    @abc.abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log-likelihood of x.

        The log-likelihood is ``log P(c) + log P(x|c)``

        """

    def predict_one(self, x):
        jll = self._joint_log_likelihood(x)
        return max(jll, key=jll.get)

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self._joint_log_likelihood(x)
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}
