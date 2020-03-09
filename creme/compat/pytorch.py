import pytest

from .. import base


__all__ = [
    'PyTorch2CremeRegressor'
]


pytest.importorskip('torch')


class PyTorch2CremeBase:

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

    def fit_one(self, x, y):

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


class PyTorch2CremeRegressor(PyTorch2CremeBase, base.Regressor):
    """PyTorch to ``creme`` regressor adapter.

    Parameters:
        net (torch.nn.Sequential)
        loss_fn (torch.nn.modules.loss._Loss)
        optimizer (torch.optim.Optimizer)
        batch_size (int)

    """

    def __init__(self, net, loss_fn, optimizer, batch_size=1):
        import torch
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=batch_size,
            x_tensor=torch.Tensor,
            y_tensor=torch.Tensor
        )

    def predict_one(self, x):
        x = self.x_tensor(list(x.values()))
        return self.net(x).item()
