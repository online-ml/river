from .base import Forecaster


class HoltWinters(Forecaster):
    def __init__(self, alpha):
        self.alpha = alpha
        self.S = None

    def learn_one(self, y):
        if self.S is None:
            self.S = y
        else:
            self.S = self.alpha * y + (1 - self.alpha) * self.S
        return self

    def forecast(self):
        return []
