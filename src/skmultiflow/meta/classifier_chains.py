import numpy as np

import copy

from sklearn.linear_model import LogisticRegression, SGDClassifier

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin
from skmultiflow.utils import check_random_state


class ClassifierChain(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin):
    """ Classifier Chains for multi-label learning.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=LogisticRegression)
        Each member of the ensemble is an instance of the base estimator

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Examples
    --------
    >>> from skmultiflow.data import make_logical
    >>>
    >>> X, Y = make_logical(random_state=1)
    >>>
    >>> print("TRUE: ")
    >>> print(Y)
    >>> print("vs")
    >>>
    >>> print("CC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    >>> print("RCC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1), order='random', random_state=1)
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    >>> print("MCC")
    >>> mcc = MCC(SGDClassifier(max_iter=100, loss='log', random_state=1), M=1000)
    >>> mcc.fit(X, Y)
    >>> Yp = mcc.predict(X, M=50)
    >>> print("with 50 iterations ...")
    >>> print(Yp)
    >>> Yp = mcc.predict(X, 'default')
    >>> print("with default (%d) iterations ..." % 1000)
    >>> print(Yp)
    >>>
    >>> print("PCC")
    >>> pcc = ProbabilisticClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
    >>> pcc.fit(X, Y)
    >>> print(pcc.predict(X))
    TRUE:
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    vs
    CC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    RCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    MCC
    with 50 iterations ...
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    with default (1000) iterations ...
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    PCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]


    Notes
    -----
    Classifier Chains [1]_ is a popular method for multi-label learning. It exploits correlation between labels
    by incrementally building binary classifiers for each label.

    scikit-learn also includes 'ClassifierChain'. A difference is probabilistic extensions are included here.


    References
    ----------
    .. [1] Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. "Classifier chains for multi-label
       classification." In Joint European Conference on Machine Learning and Knowledge Discovery in Databases,
       pp. 254-269. Springer, Berlin, Heidelberg, 2009.

    """

    # TODO: much of this can be shared with Regressor Chains, probably should use a base class to inherit here.

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
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

    def fit(self, X, y, classes=None, sample_weight=None):
        """ Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

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

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: NOT used (default=None)

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
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        N, D = X.shape
        Y = np.zeros((N, self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j-1]])
            Y[:, j] = self.ensemble[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the X entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of X belongs to a certain class-label.

        Notes
        -----
        Returns marginals [P(y_1=1|x),...,P(y_L=1|x,y_1,...,y_{L-1})]
        i.e., confidence predictions given inputs, for each instance.

        This function suitable for multi-label (binary) data
        only at the moment (may give index-out-of-bounds error if
        uni- or multi-target (of > 2 values) data is used in training).
        """
        N, D = X.shape
        Y = np.zeros((N, self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j-1]])
            Y[:, j] = self.ensemble[j].predict_proba(X)[:, 1]
        return Y

    def reset(self):
        self.__configure()
        return self

    def _more_tags(self):
        return {'multioutput': True,
                'multioutput_only': True}


def P(y, x, cc, payoff=np.prod):
    """ Payoff function, P(Y=y|X=x)

        What payoff do we get for predicting y | x, under model cc.

        Parameters
        ----------
        x: input instance
        y: its true labels
        cc: a classifier chain
        payoff: payoff function

        Returns
        -------
        A single number; the payoff of predicting y | x.
    """
    D = len(x)
    L = len(y)

    p = np.zeros(L)
    xy = np.zeros(D + L)
    xy[0:D] = x.copy()
    for j in range(L):
        P_j = cc.ensemble[j].predict_proba(xy[0:D+j].reshape(1, -1))[0]  # e.g., [0.9, 0.1] wrt 0, 1
        xy[D+j] = y[j]                                           # e.g., 1
        p[j] = P_j[y[j]]                                         # e.g., 0.1
                                                                 # or, y[j] = 0 is predicted with probability p[j] = 0.9
    return payoff(p)


class ProbabilisticClassifierChain(ClassifierChain):
    """ Probabilistic Classifier Chains for multi-label learning.

    Published as 'PCC'

    Parameters
    ----------
    base_estimator: skmultiflow or sklearn model (default=LogisticRegression)
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    """

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
        super().__init__(base_estimator=base_estimator, order=order, random_state=random_state)

    def predict(self, X):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        Notes
        -----
        Explores all possible branches of the probability tree (i.e., all possible 2^L label combinations).
        """
        N, D = X.shape
        Yp = np.zeros((N, self.L))

        # for each instance
        for n in range(N):
            w_max = 0.
            # for each and every possible label combination
            for b in range(2**self.L):
                # put together a label vector
                y_ = np.array(list(map(int, np.binary_repr(b, width=self.L))))
                # ... and gauge a probability for it (given x)
                w_ = P(y_, X[n], self)
                # if it performs well, keep it, and record the max
                if w_ > w_max:
                    Yp[n, :] = y_[:].copy()
                    w_max = w_

        return Yp


class MonteCarloClassifierChain(ProbabilisticClassifierChain):
    """ Monte Carlo Sampling Classifier Chains for multi-label learning.

        PCC, using Monte Carlo sampling, published as 'MCC'.

        M samples are taken from the posterior distribution. Therefore we need
        a probabilistic interpretation of the output, and thus, this is a
        particular variety of ProbabilisticClassifierChain.

        N.B. Multi-label (binary) only at this moment.

    Parameters
    ----------
    base_estimator: StreamModel or sklearn model
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    M: int (default=10)
        Number of samples to take from the posterior distribution.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.
    """
    def __init__(self, base_estimator=LogisticRegression(), M=10, random_state=None):
        # Do M iterations, unless overridden by M at prediction time
        ClassifierChain.__init__(self, base_estimator, random_state=random_state)
        self.M = M

    def sample(self, x):
        """
            Sample y ~ P(y|x)

            Returns
            -------
            y: a sampled label vector
            p: the associated probabilities, i.e., p(y_j=1)=p_j
        """
        D = len(x)

        p = np.zeros(self.L)
        y = np.zeros(self.L)
        xy = np.zeros(D + self.L)
        xy[0:D] = x.copy()

        for j in range(self.L):
            P_j = self.ensemble[j].predict_proba(xy[0:D + j].reshape(1, -1))[0]
            y_j = self._random_state.choice(2, 1, p=P_j)
            xy[D+j] = y_j
            y[j] = y_j
            p[j] = P_j[y_j]

        return y, p

    def predict(self, X, M=None):
        """ Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        M: int (optional, default=None)
            Number of sampling iterations. If None, M is set equal to the M value used for initialization

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        Notes
        -----
        Quite similar to the `ProbabilisticClassifierChain.predict()` function.

        Depending on the implementation, `y_max`, `w_max` may be initially set to 0,
        if we wish to rely solely on the sampling. Setting the `w_max` based on a naive CC prediction
        gives a good baseline to work from.

        """
        N, D = X.shape
        Yp = np.zeros((N, self.L))

        if M is None:
            M = self.M

        # for each instance
        for n in range(N):
            Yp[n, :] = ClassifierChain.predict(self, X[n].reshape(1, -1))
            w_max = P(Yp[n, :].astype(int), X[n], self)
            # for M times
            for m in range(M):
                y_, p_ = self.sample(X[n])  # N.B. in fact, the calculation p_ is done again in P.
                w_ = P(y_.astype(int), X[n], self)
                # if it performs well, keep it, and record the max
                if w_ > w_max:
                    Yp[n, :] = y_[:].copy()
                    w_max = w_

        return Yp
