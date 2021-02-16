import typing

import numpy as np
import pandas as pd

from river import base, optim

__all__ = ["MLPRegressor"]


def xavier_init(dims: typing.Tuple[int], seed: int = None):
    """Xavier weight initialization.

    References
    ----------
    [^1]: Glorot, X. and Bengio, Y., 2010, March. Understanding the difficulty of training deep
        feedforward neural networks. In Proceedings of the thirteenth international conference on
        artificial intelligence and statistics (pp. 249-256).

    """

    rng = np.random.RandomState(seed)

    w = {}  # weights
    b = {}  # biases

    for i in range(len(dims) - 1):
        w[i + 1] = rng.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])
        b[i + 1] = np.zeros(dims[i + 1], dtype=w[i + 1].dtype)

    return w, b


class MLP:
    """Multi-layer Perceptron.

    The `_forward` method propagates input forward through the network and stores the outputs
    of each layer along the way. The `_backward` method then goes through each layer in reverse
    order and updates the weights within each layer.

    """

    def __init__(
        self,
        hidden_dims: typing.Tuple[int],
        activations,
        loss: optim.losses.Loss,
        optimizer: optim.Optimizer,
        seed: int = None,
    ):

        self.activations = activations
        self.hidden_dims = hidden_dims
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed

    @property
    def n_layers(self) -> int:
        """Return the number of layers in the network.

        The number of layers is equal to the number of hidden layers plus 2. The 2 accounts for the
        input layer and the output layer.

        """
        return len(self.hidden_dims) + 2

    def _forward(self, X: pd.DataFrame) -> (dict, dict):
        """Execute a forward pass through the neural network.

        Parameters
        ----------
        X
            A DataFrame of shape (batch_size, n_features).

        Returns
        -------
        The output of each layer.
        The output of each activation function.

        """

        z = {}  # z = w(x) + b
        a = {}  # a = f(z)

        # As a convention, we sort the columns of the input DataFrame. This allows the user to
        # pass DataFrames with different column orders without having to worry.
        a[1] = X.values[:, np.argsort(X.columns.to_numpy())]

        for i in range(2, self.n_layers + 1):
            z[i] = np.dot(a[i - 1], self.w[i - 1]) + self.b[i - 1]
            a[i] = self.activations[i - 2].apply(z[i])

        return z, a

    def _backward(self, z: dict, a: dict, y: pd.DataFrame):
        """Execute a backward pass through the neural network, aka. backpropagation.

        Parameters
        ----------
        z
            The output of each layer.
        a
            The output of each activation function.
        y
            A DataFrame of shape (batch_size, n_targets).

        """

        y = y.values[:, np.argsort(y.columns.to_numpy())]

        # Determine the partial derivative and delta for the output layer
        y_pred = a[self.n_layers]
        final_activation = self.activations[self.n_layers - 2]
        delta = self.loss.gradient(y, y_pred) * final_activation.gradient(y_pred)
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {self.n_layers - 1: (dw, delta)}

        # Go through the layers in reverse order
        for i in range(self.n_layers - 2, 0, -1):
            delta = np.dot(delta, self.w[i + 1].T) * self.activations[i - 1].gradient(
                z[i + 1]
            )
            dw = np.dot(a[i].T, delta)
            update_params[i] = (dw, delta)

        # Update the parameters
        for i, (dw, delta) in update_params.items():
            self.optimizer.step(w=self.w[i], g=dw)
            self.optimizer.step(w=self.b[i], g=np.mean(delta, axis=0))

    def learn_many(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the network.

        Parameters
        ----------
        X
            A DataFrame of shape (batch_size, n_features).
        y
            A DataFrame of shape (batch_size, n_targets).

        """

        # We expect y to be 2D, even if there is only one target to predict
        if isinstance(y, pd.Series):
            y = y.to_frame()

        # The weights need initializing during the first call to partial_fit
        if not hasattr(self, "w"):
            self.w, self.b = xavier_init(
                dims=(X.shape[1], *self.hidden_dims, y.shape[1]), seed=self.seed
            )
            self.features = X.columns.to_numpy()
            self.features.sort()
            self.targets = y.columns.to_numpy()

        z, a = self._forward(X)
        self._backward(z, a, y)

        return self

    def __call__(self, X: pd.DataFrame):
        """Make predictions.

        Parameters
        ----------
        X
            A DataFrame of shape (batch_size, n_features).

        """
        _, a = self._forward(X)
        y_pred = a[self.n_layers]
        return pd.DataFrame(y_pred, columns=self.targets, index=X.index)


class MLPRegressor(base.Regressor, MLP):
    """Multi-layer Perceptron for regression.

    This model is still work in progress. Here are some features that still need implementing:

    - `learn_one` and `predict_one` just cast the input `dict` to a single row dataframe and then
        call `learn_many` and `predict_many` respectively. This is very inefficient.
    - Not all of the optimizers in the `optim` module can be used as they are not all vectorised.
    - Emerging and disappearing features are not supported. Each instance/batch has to have the
        same features.
    - The gradient haven't been numerically checked.

    Parameters
    ----------
    hidden_dims
        The dimensions of the hidden layers. For example, specifying `(10, 20)` means that there
        are two hidden layers with 10 and 20 neurons, respectively. Note that the number of layers
        the network contains is equal to the number of hidden layers plus two (to account for the
        input and output layers).
    activations
        The activation functions to use at each layer, including the input and output layers.
        Therefore you need to specify three activation if you specify one hidden layer.
    loss
        Loss function. Defaults to `optim.losses.Squared`.
    optimizer
        Optimizer. Defaults to `optim.SGD(.01)`.
    seed
        Random number generation seed. Set this for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import neural_net as nn
    >>> from river import optim
    >>> from river import preprocessing as pp
    >>> from river import metrics

    >>> model = (
    ...     pp.StandardScaler() |
    ...     nn.MLPRegressor(
    ...         hidden_dims=(5,),
    ...         activations=(
    ...             nn.activations.ReLU,
    ...             nn.activations.ReLU,
    ...             nn.activations.Identity
    ...         ),
    ...         optimizer=optim.SGD(1e-3),
    ...         seed=42
    ...     )
    ... )

    >>> dataset = datasets.TrumpApproval()

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.589827

    You can also use this to process mini-batches of data.

    >>> model = (
    ...     pp.StandardScaler() |
    ...     nn.MLPRegressor(
    ...         hidden_dims=(10,),
    ...         activations=(
    ...             nn.activations.ReLU,
    ...             nn.activations.ReLU,
    ...             nn.activations.ReLU
    ...         ),
    ...         optimizer=optim.SGD(1e-4),
    ...         seed=42
    ...     )
    ... )

    >>> dataset = datasets.TrumpApproval()
    >>> batch_size = 32

    >>> for epoch in range(10):
    ...     for xb in pd.read_csv(dataset.path, chunksize=batch_size):
    ...         yb = xb.pop('five_thirty_eight')
    ...         y_pred = model.predict_many(xb)
    ...         model = model.learn_many(xb, yb)

    >>> model.predict_many(xb)
          five_thirty_eight
    992           39.361609
    993           46.398536
    994           42.094086
    995           40.195802
    996           40.782954
    997           40.839678
    998           40.896403
    999           48.362659
    1000          42.021849

    """

    def __init__(
        self,
        hidden_dims,
        activations,
        loss: optim.losses.Loss = None,
        optimizer: optim.Optimizer = None,
        seed: int = None,
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            activations=activations,
            loss=loss or optim.losses.Squared(),
            optimizer=optimizer or optim.SGD(0.01),
            seed=seed,
        )

    @classmethod
    def _default_params(self):
        from . import activations

        return {
            "hidden_dims": (20,),
            "activations": (activations.ReLU, activations.ReLU, activations.Identity),
        }

    def predict_many(self, X):
        if not hasattr(self, "w"):
            return pd.DataFrame({0: 0}, index=X.index)
        return self(X)

    def learn_one(self, x, y):

        # Multi-output
        if isinstance(y, dict):
            return self.learn_many(X=pd.DataFrame([x]), y=pd.DataFrame([y]))

        # Single output
        return self.learn_many(X=pd.DataFrame([x]), y=pd.Series([y]))

    def predict_one(self, x):
        y_pred = self.predict_many(X=pd.DataFrame([x]))

        # Multi-output
        if len(y_pred.columns) > 1:
            return y_pred.iloc[0].to_dict()

        # Single output
        return y_pred.iloc[0, 0]
