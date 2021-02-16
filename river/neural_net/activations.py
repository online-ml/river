import abc

import numpy as np

__all__ = ["ReLU", "Sigmoid", "Identity"]


class Activation:
    """An activation function.

    Each activation function is represented by a class and implements two methods. The first method
    is `apply`, which evaluates the activation on an array. The second method is `gradient`, which
    computes the gradient with respect to the input array. Both methods are intended to be pure
    with no side-effects. In other words they do not modify their inputs.

    """

    @abc.abstractstaticmethod
    def apply(self, z):
        """Apply the activation function to a layer output z."""

    @abc.abstractstaticmethod
    def gradient(self, z):
        """Return the gradient with respect to a layer output z."""


class ReLU(Activation):
    """Rectified Linear Unit (ReLU) activation function."""

    @staticmethod
    def apply(z):
        a = np.copy(z)
        a[a < 0] = 0
        return a

    @staticmethod
    def gradient(z):
        a = np.zeros_like(z, dtype=z.dtype)
        a[z > 0] = 1
        return a


class Sigmoid(Activation):
    """Sigmoid activation function."""

    @staticmethod
    def apply(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.apply(z)
        return s * (1 - s)


class Identity(Activation):
    """Identity activation function."""

    @staticmethod
    def apply(z):
        return np.copy(z)

    @staticmethod
    def gradient(z):
        return np.ones_like(z, dtype=z.dtype)
