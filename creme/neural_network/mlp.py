import numpy as np


class MLP:
    """A vanilla neural network.

    The `_forward` method propagates input forward through the network and stores the outputs
    of each layer along the way. The `_backward` method then goes through each layer in reverse
    order and updates the weights within each layer.

    The parameters of each are stored in dictionaries. Each dictionary is indexed from 1 to
    n_layers - 1.

    """

    def __init__(self, dims, activations, loss, optimizer, seed=None):
        self.n_layers = len(dims)
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.w = {}  # weights
        self.b = {}  # biases
        self.activations = {}

        for i in range(len(dims) - 1):
            self.w[i + 1] = self._rng.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])  # Xavier init
            self.b[i + 1] = np.zeros(dims[i + 1], dtype=self.w[i + 1].dtype)
            self.activations[i + 2] = activations[i]

    def _forward(self, X):
        """Execute a forward pass through the neural network.

        Parameters
        ----------
        X
            An array of shape (n_samples, n_features).

        Returns
        -------
        The output of each layer.
        The output of each activation function.

        """

        z = {}  # z = w(x) + b
        a = {}  # a = f(z)

        a[1] = X  # the first layer has no activation function

        for i in range(2, self.n_layers + 1):
            z[i] = np.dot(a[i - 1], self.w[i - 1]) + self.b[i - 1]
            a[i] = self.activations[i].apply(z[i])

        return z, a

    def _backward(self, z, a, y):
        """Execute a backward pass through the neural network, aka. backpropagation.

        Parameters
        ----------
        z
            The output of each layer.
        a
            The output of each activation function.
        y
            Ground truths.

        """

        # Determine the partial derivative and delta for the output layer
        y_pred = a[self.n_layers]
        final_activation = self.activations[self.n_layers]
        delta = self.loss.gradient(y, y_pred) * final_activation.gradient(y_pred)
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # Go through the layers in reverse order
        for i in range(self.n_layers - 2, 0, -1):
            delta = np.dot(delta, self.w[i + 1].T) * self.activations[i + 1].gradient(z[i + 1])
            dw = np.dot(a[i].T, delta)
            update_params[i] = (dw, delta)

        # Update the parameters
        for i, (dw, delta) in update_params.items():
            self.optimizer.step(w=self.w[i], g=dw)
            self.optimizer.step(w=self.b[i], g=np.mean(delta, axis=0))

    def partial_fit(self, X, y):
        """Train the network.

        Parameters
        ----------
        X
            An array of shape (n_samples, n_features).
        y
            An array of shape (n_samples, n_targets). This is expected to be a 2D array, even if it
            only contains one column.

        """

        z, a = self._forward(X)
        self._backward(z, a, y)
        return self

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X
            An array of shape (n_samples, n_features).

        """

        _, a = self._forward(X)
        return a[self.n_layers]
