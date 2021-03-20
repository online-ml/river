from .. import utils
from . import base

__all__ = ["Newton"]


class Newton(base.Optimizer):
    """Online Newton Step (ONS) optimizer.

    This optimizer uses second-order information (i.e. the Hessian of the cost function) in
    addition to first-order information (i.e. the gradient of the cost function).

    Parameters
    ----------
    lr
    eps

    References
    ----------
    [^1]: [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3), pp.169-192](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)

    """

    def __init__(self, lr=0.1, eps=1e-5):
        super().__init__(lr)
        self.eps = eps
        self.H_inv = {}

    def _step(self, w, g):

        for i in g:
            if (i, i) not in self.H_inv:
                self.H_inv[i, i] = self.eps

        # Update the Hessian
        self.H = utils.math.sherman_morrison(A_inv=self.H_inv, u=g, v=g)

        # Calculate the update step
        step = utils.math.dotvecmat(x=g, A=self.H_inv)

        # Update the weights
        for i, s in step.items():
            w[i] -= self.learning_rate * s

        return w
