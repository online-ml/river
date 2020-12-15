from river import base
import torch
from vowpalwabbit import pyvw


class PyTorchModel:
    def __init__(self, network, loss_fn, optimizer):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def learn_one(self, x, y):
        x = torch.FloatTensor(list(x.values()))
        y = torch.FloatTensor([y])

        y_pred = self.network(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self


class PyTorchBinaryClassifier(PyTorchModel, base.Classifier):
    def predict_proba_one(self, x):

        x = torch.FloatTensor(list(x.values()))
        p = self.network(x).item()

        return {True: p, False: 1.0 - p}


class VW2CremeBase:
    def __init__(self, *args, **kwargs):
        self.vw = pyvw.vw(*args, **kwargs)

    def _format_x(self, x):
        return self.vw.example(" ".join(map(lambda x: ":" + str(x), x.values())))


class VW2CremeClassifier(VW2CremeBase, base.Classifier):
    def learn_one(self, x, y):

        # Convert {False, True} to {-1, 1}
        y = int(y)
        y_vw = 2 * y - 1

        ex = self._format_x(x)
        self.vw.learn(ex)
        self.vw.finish_example(ex)
        return self

    def predict_proba(self, x):
        ex = self._format_x(x)
        y_pred = self.vw.predict(ex)
        return {True: y_pred, False: 1.0 - y_pred}


class VW2RiverRegressor(VW2RiverBase, base.Regressor):
    def predict_one(self, x):
        ex = self._format_x(x)
        y_pred = self.vw.predict(ex)
        self.vw.finish_example(ex)
        return y_pred

    def learn_one(self, x):
        ex = f"{y} | {self._format_x(x)}"
        self.vw.learn(ex)
        self.vw.finish_example(ex)
        return self
