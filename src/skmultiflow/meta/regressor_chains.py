import numpy as np

import copy

from sklearn.linear_model import SGDRegressor

from skmultiflow.core import BaseSKMObject, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin
from skmultiflow.utils import check_random_state


class RegressorChain(BaseSKMObject, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin):
    """ Regressor Chains for multi-output learning.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=SGDRegressor)
        Each member of the ensemble is an instance of the base estimator.

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Notes
    -----
    Regressor Chains are a modification of Classifier Chains [1]_ for regression.

    References
    ----------
    .. [1] Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. "Classifier chains for multi-label
       classification." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases,
       pp. 254-269. Springer, Berlin, Heidelberg, 2009.
    """
    def __init__(self, base_estimator=SGDRegressor(), order=None, random_state=None):
        super().__init__()
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.chain = None
        self.ensemble = None
        self.L = None
        self._random_state = None   # This is the actual random_state object used internally
        self.__configure()

    def __configure(self):
        self.ensemble = None
        self.L = -1
        self._random_state = check_random_state(self.random_state)

    def fit(self, X, y, sample_weight=None):
        """ Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the target values of all samples in X.

        sample_weight: Not used (default=None)

        Returns
        -------
        self

        """
        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        self.chain = np.arange(L)
        if self.order == 'random':
            self._random_state.shuffle(self.chain)

        # Set the chain order
        y = y[:, self.chain]

        # Train
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(L)]
        XY = np.zeros((N, D + L-1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0:L - 1]
        for j in range(self.L):
            self.ensemble[j].fit(XY[:, 0:D + j], y[:, j])
        return self

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.

        sample_weight: Not used (default=None)

        Returns
        -------
        self

        """
        if self.ensemble is None:
            # This is the first time that the model is fit
            self.fit(X, y)
            return self

        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        # Set the chain order
        y = y[:, self.chain]

        XY = np.zeros((N, D + L-1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0:L - 1]
        for j in range(L):
            self.ensemble[j].partial_fit(XY[:, 0:D + j], y[:, j])

        return self

    def predict(self, X):
        """ Predict target values for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the target values for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        N, D = X.shape
        Y = np.zeros((N,self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j-1]])
            Y[:, j] = self.ensemble[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def reset(self):
        self.__configure()
        return self

    def predict_proba(self, X):
        """ Not implemented for this method.
        """
        raise NotImplementedError

    def _more_tags(self):
        return {'multioutput': True,
                'multioutput_only': True}
