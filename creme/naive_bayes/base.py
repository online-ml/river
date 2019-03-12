import abc

from .. import base
from .. import utils


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
        return utils.softmax(self._joint_log_likelihood(x))
