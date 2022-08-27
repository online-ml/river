from river import base


class BayesianLinearRegression(base.MiniBatchRegressor):
    """Bayesian linear regression.

    """

    def learn_one(self, x, y):
        ...

    def predict_one(self, x):
        ...

    def learn_many(self, X, y):
        ...

    def predict_many(self, X):
        ...
