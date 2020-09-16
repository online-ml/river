import collections
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
            MAE: 0.519588

            >>> model['LinearRegression'].intercept
            41.663289

    References:
        1. `Bottou, L., 2003. Stochastic Learning. Advanced Lectures on Machine Learning: ML Summer Schools 2003, pp.146-168. <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.418.7228&rep=rep1&type=pdf>`_

    """

    def __init__(self, lr=1.0):
        super().__init__(lr)
        self.H_inv = collections.defaultdict(float)


    def _update_after_pred(self, w, g, x):
        """ Kalman update
        
        For now updates only with least squares loss
        
        Comments:
        -For general losses we probably need to calculate the inverse explicitly
        -Other possibility: Woodbury matrix identity with WMI(H_inv,Eye)
        -Maybe: Decompose the hessian matrix into two vectors and use shermann-morrison ?
       
        """
        # Explicit inverse calculation for general losses
        # import numpy as np
        # increment = np.array((len(x),len(x))
        # for (i, xi), (j, xj) in x.items(), x.items():
        #     Increment[i, j] = xi * xj * self.loss.hessian(y_true=y, y_pred=self._raw_dot(x))
        # H = np.array([[self.K[i,j] for i in range(len(w))] for j in range((len(w)))])
        # H_inv = np.linalg.inv(np.linalg.inv(H_inv) + Increment)
        # for i in range(len(w)):
        #     for j in range(len(w)):
        #         self.H[i,j] = H_inv[i,j]
            
        # Update the Kalman matrix
        self.K = utils.math.sherman_morrison(A_inv=self.H_inv, u=x, v=x)

        # Calculate the update step
        step = utils.math.dotvecmat(x=g, A=self.H_inv)
        
        # Update the weights
        for i, s in step.items():
            w[i] -= self.learning_rate * s

        return w
