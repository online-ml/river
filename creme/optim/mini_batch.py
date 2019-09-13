import collections

from . import base


class MiniBatcher(base.Optimizer):
    """Wrapper class for doing mini-batch gradient descent.

    This will accumulate successive gradients until ``batch_size`` have been observed. Once the
    current mini-batch is full, the underlying optimizer will update the weights using the average
    gradient. This can significantly improve the accuracy of a model, especially early on when
    gradients tend to go wild.

    Parameters:
        optimizer (optim.Optimizer)
        batch_size (int): The desired batch size.

    Attributes:
        gradient (collections.defaultdict): The current accumulated mini-batch.
        current_size (int): The current size of the mini-batch.
        n_iterations (int)

    Example:

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> optimizer = optim.MiniBatcher(optim.SGD(0.1), 4)
            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     linear_model.LogisticRegression(optimizer)
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.969014

    """

    def __init__(self, optimizer, batch_size):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gradient = collections.defaultdict(float)
        self.current_size = 0
        self.n_iterations = 0

    def update_before_pred(self, w):
        return self.optimizer.update_before_pred(w)

    def _update_after_pred(self, w, g):

        # Accumulate the gradient
        for i, gi in g.items():
            self.gradient[i] += gi
        self.current_size += 1

        if self.current_size == self.batch_size:

            # Update the weights with the average gradient
            avg_g = {i: gi / self.batch_size for i, gi in self.gradient.items()}
            w = self.optimizer.update_after_pred(w, avg_g)

            # Reset the gradient
            self.gradient = collections.defaultdict(float)
            self.current_size = 0

        return w
