from __future__ import annotations

import numpy as np


class AUC_SGD:
    """
        AUC Stochastic Gradient Descent (SGD)

        This class implements an SGD-based optimization method for maximizing the AUC (Area Under the Curve)
        of a binary classifier assuming a linear regression model.

        Attributes
        ----------
        epochs : int
            Number of training epochs.
        lr : float
            Initial learning rate for gradient descent updates.
        n_mc : int
            Number of Monte Carlo samples used for estimating gradients.
        gamma : float
            Learning rate decay parameter.
        eps : float
            Smoothing parameter for numerical stability.

        Methods
        -------

        getTrain(X_train, y_train):
            Returns the Prediction to maximize training AUC score.
        getTest(X_test, y_test):
            Returns the Prediction to maximize testing AUC score.

        Examples
        --------
        >>> from river import linear_model
        >>> from sklearn.metrics import roc_auc_score
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(n_samples=2000, n_informative=9, n_redundant=0, n_repeated=0, random_state=2)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=4)
        >>> base = LogisticRegression().fit(X_train, y_train)
        >>> X1 = X_train[y_train==1]
        >>> X0 = X_train[y_train==0]
        >>> model = linear_model.AUC_SGD()
        >>> np.random.seed(123)
        >>> theta = np.random.randn(X_train[0].shape[0])
        >>> test_auc = model.getTest(X_train, X_test, y_train)
        >>> train_auc = model.getTrain(X_train, y_train)
        >>> print(roc_auc_score(y_train, train_auc))
        0.8899135830932864
        >>> print(roc_auc_score(y_test, test_auc))
        0.8849634963496349
        """

    def __init__(self, epochs=900, lr=0.5, n_mc=500, gamma=1e-4, eps=0.01):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.n_mc = n_mc
        self.gamma = gamma
        self.eps = eps

    def sigma_eps(self, z, eps):
        q = - z / eps
        if abs(q) < 35:
            return 1 / (1 + np.exp(q))
        elif q > 0:
            return 0
        else:
            return 1


    def stochastic_gradient(self, theta, X1, X0, N=1000, eps=0.01, random_state=1):
        """
        Computes the stochastic gradient of the AUC objective.

        This function calculates the gradient of the AUC metric with respect to the parameter `theta` using
        Monte Carlo sampling. Positive and negative samples are drawn to estimate the difference in predictions.

        Parameters
            ----------
        theta : numpy.ndarray
            The parameter vector (weights) of the model.
        X1 : numpy.ndarray
            Feature matrix for the positive class (label = 1).
        X0 : numpy.ndarray
            Feature matrix for the negative class (label = 0).
        N : int, optional, default=1000
            Number of Monte Carlo samples to use for gradient estimation.
        eps : float, optional, default=0.01
            Smoothing parameter for the sigmoid function.
        random_state : int, optional, default=1
            Random seed for reproducibility.

        Returns
        -------
        numpy.ndarray
            The estimated gradient vector.
        """

        np.random.seed(random_state)
        indices_1 = np.random.choice(np.arange(X1.shape[0]), size=N)
        indices_0 = np.random.choice(np.arange(X0.shape[0]), size=N)
        X1_, X0_ = X1[indices_1], X0[indices_0]
        avg = np.zeros_like(theta)

        for xi, xj in zip(X1_, X0_):
            dx = xj - xi
            sig = self.sigma_eps(np.dot(theta, dx), eps)
            avg += sig * (1 - sig) * dx
        return avg / (N * eps)

    def compute(self, X_train, X_test, y_train):
        X1 = X_train[y_train == 1]
        X0 = X_train[y_train == 0]
        np.random.seed(123)
        theta = np.random.randn(X_train[0].shape[0])
        epochs_list = list(range(self.epochs))

        for seed, epoch in enumerate(epochs_list):
            # learning rate scheduler
            self.lr = self.lr / (1 + self.gamma)

            theta = theta - self.lr * self.stochastic_gradient(theta, X1, X0, N=self.n_mc, random_state=seed)

        if X_test is not None:
            return theta @ X_test.T
        else:
            return theta @ X_train.T

    def getTrain(self, X_train, y_train):
        """
        Implements the stochastic gradient ascent method to optimize theta for a maximized AUC training score.

        Parameters:
        - X_train: Training feature matrix.
        - y_train: Training labels.

        Returns:
        - Prediction to maximize training AUC score.
        """

        return self.compute(X_train, None, y_train)

    def getTest(self, X_train, X_test, y_train):
        """
        Implements the stochastic gradient ascent method to optimize theta for a maximized AUC testing score.

        Parameters:
        - X_train: Training feature matrix.
        - X_test: Testing feature matrix.
        - y_train: Training labels.

        Returns:
        - Prediction to maximize testing AUC score.
        """
        return self.compute(X_train, X_test, y_train)
