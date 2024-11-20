import numpy as np

def sigma_eps(z, eps):
    return 1 / (1 + np.exp(-z / eps))


class AUCMetric:
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

    def get(self, X_train, y_train, X_test, y_test, epochs=900, lr=0.5, n_mc=500, gamma=1e-4, eps=0.01):
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

        # Define the stochastic gradient function
    def stochastic_gradient(theta, X1, X0, N, eps, random_state=1):
        np.random.seed(random_state)
        indices_1 = np.random.choice(np.arange(X1.shape[0]), size=N)
        indices_0 = np.random.choice(np.arange(X0.shape[0]), size=N)
        X1_, X0_ = X1[indices_1], X0[indices_0]
        avg = np.zeros_like(theta)
        for xi, xj in zip(X1_, X0_):
            dx = xj - xi
            sig = sigma_eps(np.dot(theta, dx))
            avg += sig * (1 - sig) * dx
        return avg / (N * eps)

