import math

from . import base


class CrossEntropy(base.MultiClassificationLoss):
    """Cross entropy is a generalization of logistic loss to multiple classes.

    Example:

        ::

            >>> from creme import optim

            >>> y_true = [0, 1, 2, 2]
            >>> y_pred = [
            ...     {0: 0.29450637, 1: 0.34216758, 2: 0.36332605},
            ...     {0: 0.21290077, 1: 0.32728332, 2: 0.45981591},
            ...     {0: 0.42860913, 1: 0.33380113, 2: 0.23758974},
            ...     {0: 0.44941979, 1: 0.32962558, 2: 0.22095463}
            ... ]

            >>> loss = optim.CrossEntropy()

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss(y_t, y_p))
            1.222454...
            1.116929...
            1.437209...
            1.509797...

            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(loss.gradient(y_t, y_p))
            {0: -0.70549363, 1: 0.34216758, 2: 0.36332605}
            {0: 0.21290077, 1: -0.67271668, 2: 0.45981591}
            {0: 0.42860913, 1: 0.33380113, 2: -0.76241026}
            {0: 0.44941979, 1: 0.32962558, 2: -0.77904537}

    References:
        1. `What is Softmax regression and how is it related to Logistic regression? <https://github.com/rasbt/python-machine-learning-book/blob/master/faq/softmax_regression.md>`_

    """

    def __call__(self, y_true, y_pred):
        return -sum(
            (y_true == label) * math.log(self.clamp_proba(proba))
            for label, proba in y_pred.items()
        )

    def gradient(self, y_true, y_pred):
        return {
            label: self.clamp_proba(y_pred.get(label, 0.)) - (y_true == label)
            for label in {*y_pred.keys(), y_true}
        }
