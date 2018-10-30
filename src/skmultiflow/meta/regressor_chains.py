import numpy as np
import copy
from skmultiflow.core.base import StreamModel
from sklearn.linear_model import SGDRegressor
from skmultiflow.utils import check_random_state


class RegressorChain(StreamModel):
    """ Classifier Chains for multi-label learning.

    Parameters
    ----------
    base_estimator: StreamModel or sklearn model
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    order : str
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
        self.chain = None
        self.ensemble = None
        self.L = None
        self._init_random_state = random_state
        self.__configure()

    def __configure(self):
        self.ensemble = None
        self.L = -1
        self.random_state = check_random_state(self._init_random_state)

    def fit(self, X, Y):
        """ fit
        """
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        self.chain = np.arange(L)
        if self.order == 'random':
            self.random_state.shuffle(self.chain)

        # Set the chain order
        Y = Y[:, self.chain]

        # Train
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(L)]
        XY = np.zeros((N, D + L-1))
        XY[:, 0:D] = X
        XY[:, D:] = Y[:, 0:L-1]
        for j in range(self.L):
            self.ensemble[j].fit(XY[:, 0:D + j], Y[:, j])
        return self

    def partial_fit(self, X, Y):
        """ partial_fit

            N.B. Assume that fit has already been called
            (i.e., this is more of an 'update')
        """
        if self.ensemble is None:
            # This was not the first time that the model is fit
            self.fit(X, Y)
            return self

        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        # Set the chain order
        Y = Y[:, self.chain]

        XY = np.zeros((N, D + L-1))
        XY[:, 0:D] = X
        XY[:, D:] = Y[:, 0:L-1]
        for j in range(L):
            self.ensemble[j].partial_fit(XY[:, 0:D + j], Y[:, j])

        return self

    def predict(self, X):
        """ predict

            Returns predictions for X
        """
        N, D = X.shape
        Y = np.zeros((N,self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j-1]])
            Y[:, j] = self.ensemble[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        self.__configure()

    def get_info(self):
        return 'RegressorChain estimator:' \
               ' - base_estimator: {}'.format(self.base_estimator) + \
               ' - order: {}'.format(self.order) + \
               ' - random_state: {}'.format(self._init_random_state)

    def predict_proba(self, X):
        raise NotImplementedError
