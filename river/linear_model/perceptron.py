from __future__ import annotations

from river import optim

from .log_reg import LogisticRegression


class Perceptron(LogisticRegression):
    """Perceptron classifier.

    In this implementation, the Perceptron is viewed as a special case of the logistic regression.
    The loss function that is used is the Hinge loss with a threshold set to 0, whilst the learning
    rate of the stochastic gradient descent procedure is set to 1 for both the weights and the
    intercept.

    Parameters
    ----------
    l2
        Amount of L2 regularization used to push weights towards 0.
    clip_gradient
        Clips the absolute value of each gradient value.
    initializer
        Weights initialization scheme.

    Attributes
    ----------
    weights
        The current weights.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model as lm
    >>> from river import metrics
    >>> from river import preprocessing as pp

    >>> dataset = datasets.Phishing()

    >>> model = pp.StandardScaler() | lm.Perceptron()

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 85.84%

    """

    def __init__(
        self,
        l2=0.0,
        clip_gradient=1e12,
        initializer: optim.initializers.Initializer | None = None,
    ):
        super().__init__(
            optimizer=optim.SGD(1),
            intercept_lr=1,
            loss=optim.losses.Hinge(threshold=0.0),
            l2=l2,
            clip_gradient=clip_gradient,
            initializer=initializer,
        )
