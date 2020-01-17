import collections
import numpy as np

from . import base


__all__ = ['Nadam']


class Nadam(base.Optimizer):
    """Nadam optimizer.

    Example:
    """

    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = collections.defaultdict(float)
        self.v = collections.defaultdict(float)

    def _update_after_pred(self, w, g):

        # Correct bias for `v`
        lr = self.learning_rate * \
            (1 - self.beta_2 ** (self.n_iterations + 1)) ** .5
        # Correct bias for `m`
        lr /= (1 - self.beta_1 ** (self.n_iterations + 1))

        for i, gi in g.items():
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gi
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * gi ** 2

            w[i] -= lr * (self.beta_1*self.m[i] + (1-self.beta_1)*gi*(1-np.power(
                self.beta_1, self.n_iterations+1))) / (self.v[i] ** .5 + self.eps)

        return w
