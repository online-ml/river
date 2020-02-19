import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils import FastBuffer, get_dimensions


class WindowedStandardScaler(StreamTransform):
    """ Standardize features by removing the mean and scaling to unit variance.
    Mean and stdev are computed in given window frame.

    Parameters
    ----------
    window_size: int (Default: 200)
        Defines the window size to compute mean and standard deviation.

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
                mean = self._get_mean(j)
                stdev = self._get_std(j)
                transformed = (value - mean) / stdev
                X[i][j] = transformed
            self.window.add_element(row)
        return X

    def _get_mean(self, column_index):
        mean = 0.
        if not self.window.is_empty():
            mean = np.nanmean(np.array(self.window.get_queue())[:, column_index])
        return mean

    def _get_std(self, column_index):
        std = 1.
        if not self.window.is_empty():
            std = np.nanstd(np.array(self.window.get_queue())[:, column_index])
        if(std == 0.):
            std = 1.
        return std

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
        StandardScaler
            self

        """
        self.window.add_element(X)
        return self
