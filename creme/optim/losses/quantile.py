class QuantileLoss:
    """Quantile loss.

    Parameters:
        alpha (float): Desired quantile to attain.

    Example:

        ::

            >>> from creme import optim

            >>> loss = optim.QuantileLoss(0.5)
            >>> loss(1, 3)
            1.0

            >>> loss.gradient(1, 3)
            0.5

            >>> loss.gradient(3, 1)
            -0.5

    References:
        1. `Wikipedia article on quantile regression <https://www.wikiwand.com/en/Quantile_regression>`_
        2. `Derivative from WolframAlpha <https://www.wolframalpha.com/input/?i=derivative+(y+-+p)+*+(alpha+-+Boole(y+-+p))+wrt+p>`_

    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        diff = y_pred - y_true
        return (self.alpha - (diff < 0)) * diff

    def gradient(self, y_true, y_pred):
        return (y_true < y_pred) - self.alpha
