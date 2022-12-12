from river import base
import torch

class PyTorchLogReg(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class PyTorchModel:
    def __init__(self, network_func, loss, optimizer_func):
        self.network_func = network_func
        self.loss = loss
        self.optimizer_func = optimizer_func

        self.network = None
        self.optimizer = None

    def learn_one(self, x, y):

        # We only know how many features a dataset contains at runtime
        if self.network is None:
            self.network = self.network_func(n_features=len(x))
            self.optimizer = self.optimizer_func(self.network.parameters())

        x = torch.FloatTensor(list(x.values()))
        y = torch.FloatTensor([y])

        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self


class PyTorchBinaryClassifier(PyTorchModel, base.Classifier):
    def predict_proba_one(self, x):

        # We only know how many features a dataset contains at runtime
        if self.network is None:
            self.network = self.network_func(n_features=len(x))
            self.optimizer = self.optimizer_func(self.network.parameters())

        x = torch.FloatTensor(list(x.values()))
        p = self.network(x).item()
        return {True: p, False: 1.0 - p}