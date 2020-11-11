import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils import FastBuffer, get_dimensions


class WindowedMinmaxScaler(StreamTransform):
    """ Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.
    For the training set we consider a window of a given length.

    Parameters
    ----------
    window_size: int (Default: 200)
        Defines the window size to compute min and max values.

    Examples
    --------
    """

    def __init__(self, window_size=200):
        super().__init__()
        self.window_size = window_size
        self.window = None

        self.__configure()

    def __configure(self):
        self.window = FastBuffer(max_size=self.window_size)

    def transform(self, X):
        """ Does the transformation process in the samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        """
        r, c = get_dimensions(X)
        for i in range(r):
            row = np.copy([X[i][:]])
            for j in range(c):
                value = X[i][j]
                min_val = self._get_min(j)
                max_val = self._get_max(j)
                if((max_val-min_val)==0):
                    transformed=0
                else:
                    X_std = (value - min_val) / (max_val - min_val)
                    transformed = X_std * (max_val - min_val) + min_val
                X[i][j] = transformed
            self.window.add_element(row)
        return X

    def _get_min(self, column_index):
        min_val = 0.
        if not self.window.is_empty():
            min_val = np.nanmin(np.array(self.window.get_queue())[:, column_index])
        return min_val

    def _get_max(self, column_index):
        max_val = 1.
        if not self.window.is_empty():
            max_val = np.nanmax(np.array(self.window.get_queue())[:, column_index])
        return max_val

    def partial_fit_transform(self, X, y=None):
        """ Partially fits the model and then apply the transform to the data.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        y: numpy.ndarray (optional, default=None)
            The target values.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            The transformed data.

        """
        X = self.transform(X)

        return X

    def partial_fit(self, X, y=None):
        """ Partial fits the model.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        y: numpy.ndarray (optional, default=None)
            The target values.

        Returns
        -------
        MinmaxScaler
            self

        """
        self.window.add_element(X)
        return self
