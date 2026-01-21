from __future__ import annotations

import numpy as np

from river import metrics


class IncrementalAUC(metrics.base.BinaryMetric):
    """Calculates AUC incrementally."""

    def __init__(self):
        super().__init__()
        self.positive_scores = []
        self.negative_scores = []

    def update(self, y_true, y_pred):
        """Updates the metric with the new prediction and true label."""
        if y_true == 1:
            self.positive_scores.append(y_pred)
        else:
            self.negative_scores.append(y_pred)
        return self

    def get(
        self, X_train, y_train, X_test, y_test, epochs=900, lr=0.5, n_mc=500, gamma=1e-4, eps=0.01
    ):
        """
        Implements the stochastic gradient ascent method to optimize theta and computes the AUC
        based on the accumulated scores.

        Parameters:
        - X_train: Training feature matrix.
        - y_train: Training labels.
        - X_test: Test feature matrix.
        - y_test: Test labels.
        - epochs: Number of training epochs.
        - lr: Initial learning rate.
        - n_mc: Number of Monte Carlo samples for gradient estimation.
        - gamma: Learning rate discount factor.
        - eps: Smoothing parameter for the sigmoid function.

        Returns:
        - auc: Final AUC score based on the accumulated scores.
        """
        from sklearn.metrics import roc_auc_score

        # Separate the classes
        X1 = X_train[y_train == 1]
        X0 = X_train[y_train == 0]

        # Initialize parameters
        np.random.seed(123)
        theta = np.random.randn(X_train.shape[1])
        current_lr = lr

        # Reset accumulated scores
        self.positive_scores = []
        self.negative_scores = []

        # Optimization loop
        for seed, epoch in enumerate(range(epochs)):
            # Update learning rate
            current_lr = current_lr / (1 + gamma)

            # Update theta using stochastic gradient ascent
            theta -= current_lr * self.stochastic_gradient(
                theta, X1, X0, N=n_mc, eps=eps, random_state=seed
            )

        # After training, compute the scores on the test set
        y_scores = np.dot(X_test, theta)

        # Update accumulated scores using y_test and y_scores
        for y_true_sample, y_score in zip(y_test, y_scores):
            self.update(y_true_sample, y_score)

        # Compute AUC based on accumulated scores
        y_scores_accumulated = self.positive_scores + self.negative_scores
        y_true_accumulated = [1] * len(self.positive_scores) + [0] * len(self.negative_scores)

        auc = roc_auc_score(y_true_accumulated, y_scores_accumulated)
        return auc

    def sigma_eps(self, x, eps=0.01):
        z = x / eps
        if z > 35:
            return 1
        elif z < -35:
            return 0
        else:
            return 1.0 / (1.0 + np.exp(-z))

    def reg_u_statistic(self, y_true, y_probs, eps=0.01):
        p = y_probs[y_true == 1]
        q = y_probs[y_true == 0]

        aux = []
        for pp in p:
            for qq in q:
                aux.append(self.sigma_eps(pp - qq, eps=eps))

        u = np.array(aux).mean()
        return u

    def stochastic_gradient(self, theta, X1, X0, N=1000, eps=0.01, random_state=1):
        np.random.seed(random_state)

        indices_1 = np.random.choice(np.arange(X1.shape[0]), size=N)
        indices_0 = np.random.choice(np.arange(X0.shape[0]), size=N)

        X1_, X0_ = X1[indices_1], X0[indices_0]

        avg = np.zeros_like(theta)
        for xi, xj in zip(X1_, X0_):
            dx = xj - xi
            sig = self.sigma_eps(theta @ dx, eps=eps)
            avg = avg + sig * (1 - sig) * dx

        return avg / (N * eps)
