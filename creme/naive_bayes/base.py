import abc
import math

from scipy import special

from creme import base


class BaseNB(base.MultiClassifier, abc.ABC):
    """Base Naive Bayes class."""

    @abc.abstractmethod
    def joint_log_likelihood(self, x: dict) -> float:
        """Compute the unnormalized posterior log-likelihood of x.

        The log-likelihood is `log P(c) + log P(x|c)`.

        """

    def predict_proba_one(self, x):
        """Return probabilities using the log-likelihoods."""
        jll = self.joint_log_likelihood(x)
        if not jll:
            return {}
        lse = special.logsumexp(list(jll.values()))
        return {label: math.exp(ll - lse) for label, ll in jll.items()}
