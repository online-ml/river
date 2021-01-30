import torch

from .. import base

__all__ = ["PyTorch2RiverRegressor"]


class PyTorch2RiverBase:
    def __init__(self, net, loss_fn, optimizer, batch_size, x_tensor, y_tensor):
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self._x_batch = [None] * batch_size
        self._y_batch = [None] * batch_size
        self._batch_i = 0

    def learn_one(self, x, y):

        self._x_batch[self._batch_i] = list(x.values())
        self._y_batch[self._batch_i] = [y]

        self._batch_i += 1

        if self._batch_i == self.batch_size:

            x = self.x_tensor(self._x_batch)
            y = self.y_tensor(self._y_batch)

            self.optimizer.zero_grad()
            y_pred = self.net(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self._batch_i = 0

        return self


class PyTorch2RiverRegressor(PyTorch2RiverBase, base.Regressor):
    """Compatibility layer from PyTorch to River for regression.

    Parameters
    ----------
    net
    loss_fn
    optimizer
    batch_size

    Examples
    --------

    >>> from river import compat
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> import torch
    >>> from torch import nn
    >>> from torch import optim

    >>> _ = torch.manual_seed(0)

    >>> dataset = datasets.TrumpApproval()

    >>> n_features = 6
    >>> net = nn.Sequential(
    ...     nn.Linear(n_features, 3),
    ...     nn.Linear(3, 1)
    ... )

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     compat.PyTorch2RiverRegressor(
    ...         net=net,
    ...         loss_fn=nn.MSELoss(),
    ...         optimizer=optim.SGD(net.parameters(), lr=1e-3),
    ...         batch_size=2
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric).get()
    2.78258

    """

    def __init__(
        self,
        net: torch.nn.Sequential,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        batch_size=1,
    ):
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=batch_size,
            x_tensor=torch.Tensor,
            y_tensor=torch.Tensor,
        )

    def predict_one(self, x):
        x = self.x_tensor(list(x.values()))
        return self.net(x).item()
