from river import base
from sklearn import exceptions
import torch
from vowpalwabbit import pyvw


class ScikitLearnClassifier(base.Classifier):

    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        self.model.partial_fit([list(x.values())], [y], classes=self.classes)
        return self

    def predict_proba_one(self, x):
        try:
            return dict(zip(self.classes, self.model.predict_proba([list(x.values())])[0]))
        except exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_one(self, x):
        try:
            return self.model.predict([list(x.values())])[0]
        except exceptions.NotFittedError:
            return self.classes[0]


class ScikitLearnRegressor(base.Regressor):

    def __init__(self, model):
        self.model = model

    def learn_one(self, x, y):
        self.model.partial_fit([list(x.values())], [y])
        return self

    def predict_one(self, x):
        try:
            return self.model.predict([list(x.values())])[0]
        except exceptions.NotFittedError:
            return 0


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


class PyTorchRegressor(PyTorchModel, base.Regressor):

    def predict_one(self, x):
        x = torch.FloatTensor(list(x.values()))
        return self.network(x).item()


class PyTorchBinaryClassifier(PyTorchModel, base.Classifier):

    def predict_proba_one(self, x):

        x = torch.FloatTensor(list(x.values()))
        p = self.network(x).item()

        return {True: p, False: 1. - p}


class KerasModel:

    def __init__(self, model):
        self.model = model

    def learn_one(self, x, y):
        x = [[list(x.values())]]
        y = [[y]]
        self.model.train_on_batch(x, y)
        return self


class KerasRegressor(KerasModel, base.Regressor):

    def predict_one(self, x):
        x = [[list(x.values())]]
        return self.model.predict_on_batch(x)[0][0]


class KerasBinaryClassifier(KerasModel, base.Classifier):

    def predict_proba_one(self, x):
        x = [[list(x.values())]]
        p_true = self.model.predict_on_batch(x)[0][0]
        return {True: p_true, False: 1. - p_true}


class VW2CremeBase:

    def __init__(self, *args, **kwargs):
        self.vw = pyvw.vw(*args, **kwargs)

    def _format_x(self, x):
        return self.vw.example(' '.join(map(lambda x: ':' + str(x), x.values())))


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
        return {True: y_pred, False: 1. - y_pred}


class VW2CremeRegressor(VW2CremeBase, base.Regressor):

    def predict_one(self, x):
        ex = self._format_x(x)
        self.vw.learn(ex)
        self.vw.finish_example(ex)
        return self
