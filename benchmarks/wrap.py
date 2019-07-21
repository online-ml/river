from creme import base
from sklearn import exceptions
import torch
from vowpalwabbit import pyvw


class ScikitLearnRegressor(base.Regressor):

    def __init__(self, model):
        self.model = model

    def fit_one(self, x, y):
        self.model.partial_fit([list(x.values())], [y])
        return self

    def predict_one(self, x):
        try:
            return self.model.predict([list(x.values())])[0]
        except exceptions.NotFittedError:
            return 0


class PyTorchRegressor(base.Regressor):

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit_one(self, x, y):

        x = torch.FloatTensor(list(x.values()))
        y = torch.FloatTensor([y])

        y_pred = self.model(x)
        loss = self.loss_fn(y, y_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self

    def predict_one(self, x):
        x = torch.FloatTensor(list(x.values()))
        return self.model(x).item()


class VowpalWabbitRegressor(base.Regressor):

    def __init__(self, **kwargs):
        kwargs['passes'] = 1
        self.model = pyvw.vw('--quiet', **kwargs)

    def format_features(self, x):
        return ' '.join((f'{k}:{v}' for k, v in x.items()))

    def fit_one(self, x, y):
        self.model.learn(f'{y} | {self.format_features(x)}')
        return self

    def predict_one(self, x):
        return self.model.predict(self.format_features(x))
