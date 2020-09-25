from . import base
from .. import utils


__all__ = ['KalmanFilter']


class KalmanFilter(base.Optimizer):
    """Kalman filter optimizer.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import preprocessing
            >>> from creme import optim

            >>> X_y = datasets.TrumpApproval()
            >>> optimizer = optim.KalmanFilter()
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LinearRegression(intercept_lr=.1,optimizer=optimizer)
            ... )
            >>> metric = metrics.MAE()

            >>> model_selection.progressive_val_score(X_y, model, metric)
            MAE: 0.513453

            >>> model['LinearRegression'].intercept
            41.625274

    References:
        1. `Bottou, L., 2003. Stochastic Learning. Advanced Lectures on Machine Learning: ML Summer Schools 2003, pp.146-168. <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.418.7228&rep=rep1&type=pdf>`_

    """

    def __init__(self, lr=1.0, eps=1e-5):
        super().__init__(lr)
        self.K = {}
        self.eps = eps


    def _update_after_pred(self, w, g, H):

        for i in g:
            if (i, i) not in self.K:
                self.K[i, i] = self.eps
        
        # Update Kalman matrix
        self.K = utils.math.woodbury_identity(A_inv=self.K, U=utils.math.eye_like(H), V=H)
        
        # Calculate the update step
        step = utils.math.dotvecmat(x=g, A=self.K)
        
        # Update the weights
        for i, s in step.items():
            w[i] -= self.learning_rate * s

        return w
