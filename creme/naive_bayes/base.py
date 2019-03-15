import abc
import math

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

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jlh = self._joint_log_likelihood(x)
        lse = math.log(sum(math.exp(l) for l in jlh.values()) or 1)
        return {label: l - lse for label, l in jlh.items()}
