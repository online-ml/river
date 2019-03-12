import abc
import typing

from . import lr_schedule


class Optimizer(abc.ABC):

    def __init__(self, lr: typing.Union[lr_schedule.LRScheduler, float]):
        self.lr = lr_schedule.ConstantLR(lr) if isinstance(lr, float) else lr
        self.n_iterations = 0

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    def update_weights_with_gradient(self, w: dict, g: dict) -> dict:
        raise NotImplementedError

    def update_weights(self, x, y, w, loss, f_pred, f_grad):
        """

        This method gives more control over the gradient computation than
        `update_weights_with_gradient`. This is required for optimisation methods that have to
        take a step before actually computing the gradient, such as FTRL-Proximal and Nesterov
        Momentum. However most optimizers could (and should) directly use
        `update_weights_with_gradient`.

        """

        # Predict the output of the given features
        y_pred = f_pred(x, w)

        # Compute the gradient
        gradient = f_grad(y_true=y, y_pred=y_pred, loss=loss, x=x, w=w)

        # Update the weights
        w = self.update_weights_with_gradient(w, gradient)

        # Update the iteration counter
        self.n_iterations += 1

        return w, y_pred
