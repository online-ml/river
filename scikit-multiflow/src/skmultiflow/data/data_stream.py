import pandas as pd
import numpy as np

import warnings

from skmultiflow.data.base_stream import Stream


class DataStream(Stream):
    """ Creates a stream from a data source.

    DataStream takes the whole data set containing the `X` (features) and `Y` (targets)
    or takes `X` and `Y` separately. For the first case `target_idx` and `n_targets` need to
    be provided, in the second case they are not needed.

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame (Default=None)
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    y: np.ndarray or pd.DataFrame, optional (Default=None)
        The targets' columns.

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    name: str, optional (default=None)
        A string to id the data.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    Notes
    -----
    The stream object provides upon request a number of samples, in a way such that old samples
    cannot be accessed at a later time. This is done to correctly simulate the stream context.

    """

    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'
    _Y_is_defined = False

    def __init__(self, data, y=None, target_idx=-1, n_targets=1, cat_features=None, name=None,
                 allow_nan=False):
        super().__init__()
        self.X = None
        self.y = y
        self.cat_features = cat_features
        self.cat_features_idx = [] if self.cat_features is None else self.cat_features
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.task_type = None
        self.n_classes = 0
        self.data = data
        self._is_ready = False
        self.name = name
        self.allow_nan = allow_nan

        if self._Y_is_defined:
            self.y = pd.DataFrame(self.y)
            if self.y.shape[0] != self.data.shape[0]:
                raise ValueError("X and y should have the same number of rows")
            else:
                self.X = pd.DataFrame(self.data)
                self.target_idx = -self.y.shape[1]
                self.n_targets = self.y.shape[1]

        self._prepare_for_use()

    @property
    def y(self):
        """
        Return the targets' columns.

        Returns
        -------
        np.ndarray:
            the targets' columns
        """
        return self._y

    @y.setter
    def y(self, y):
        """
        Sets the targets' columns

        Parameters
        ----------
        y: pd.DataFrame or np.ndarray
            the targets' columns

        """
        if y is not None and not self._Y_is_defined:
            self._Y_is_defined = True
        if not self._Y_is_defined or (isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame)):
            self._y = y
        else:
            raise ValueError(
                "np.ndarray or pd.DataFrame y object expected, and {} was passed".format(type(y)))

    @property
    def X(self):
        """
        Return the features' columns.

        Returns
        -------
        np.ndarray:
            the features' columns
        """
        return self._X

    @X.setter
    def X(self, X):
        """
        Sets the features' columns.

        Parameters
        ----------
        X: pd.DataFrame or np.ndarray
            the features' columns.
        """

        if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or not self._Y_is_defined:
            self._X = X

        else:
            raise ValueError(
                "np.ndarray or pd.DataFrame X object expected, and {} was passed".format(type(X)))

    @property
    def data(self):
        """
        Return the data set used to generate the stream.

        Returns
        -------
        pd.DataFrame:
            Data set.
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Sets the data set used to generate the stream.

        Parameters
        ----------
        data: DataFrame or np.ndarray
            the data set

        """

        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = pd.DataFrame(data)
        else:
            raise ValueError("Invalid type {}, for data".format(type(data)))

    @data.deleter
    def data(self):
        """
            Deletes data
        """
        del self._data

    @property
    def target_idx(self):
        """
        Get the number of the column where Y begins.

        Returns
        -------
        int:
            The number of the column where Y begins.
        """
        return self._target_idx

    @target_idx.setter
    def target_idx(self, target_idx):
        """
        Sets the number of the column where Y begins.

        Parameters
        ----------
        target_idx: int
        """

        self._target_idx = target_idx

    @property
    def n_targets(self):
        """
         Get the number of targets.

        Returns
        -------
        int:
            The number of targets.
        """
        return self._n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        """
        Sets the number of targets.

        Parameters
        ----------
        n_targets: int
        """

        self._n_targets = n_targets

    @property
    def cat_features_idx(self):
        """
        Get the list of the categorical features index.

        Returns
        -------
        list:
            List of categorical features index.

        """
        return self._cat_features_idx

    @cat_features_idx.setter
    def cat_features_idx(self, cat_features_idx):
        """
        Sets the list of the categorical features index.

        Parameters
        ----------
        cat_features_idx:
            List of categorical features index.
        """

        self._cat_features_idx = cat_features_idx

    def _prepare_for_use(self):
        self.restart()
        if not self._is_ready:
            if self._Y_is_defined:
                self._load_X_y()
            else:
                self._load_data()
                del self.data
            self._is_ready = True

    def _load_X_y(self):

        self.y = pd.DataFrame(self.y)

        check_data_consistency(self.y, self.allow_nan)
        check_data_consistency(self.X, self.allow_nan)

        self.n_samples, self.n_features = self.X.shape
        self.feature_names = self.X.columns.values.tolist()
        self.target_names = self.y.columns.values.tolist()

        self.y = self.y.values
        self.X = self.X.values

        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} '
                                 'exceeds n_features {}'.format(self.cat_features_idx,
                                                                self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if np.issubdtype(self.y.dtype, np.integer):
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self._REGRESSION

        self.target_values = self._get_target_values()

    def _load_data(self):

        check_data_consistency(self.data, self.allow_nan)

        rows, cols = self.data.shape
        self.n_samples = rows
        labels = self.data.columns.values.tolist()

        if (self.target_idx + self.n_targets) == cols or (self.target_idx + self.n_targets) == 0:
            # Take everything to the right of target_idx
            self.y = self.data.iloc[:, self.target_idx:].values
            self.target_names = self.data.iloc[:, self.target_idx:].columns.values.tolist()
        else:
            # Take only n_targets columns to the right of target_idx, use the rest as features
            self.y = self.data.iloc[:, self.target_idx:self.target_idx + self.n_targets].values
            self.target_names = labels[self.target_idx:self.target_idx + self.n_targets]

        self.X = self.data.drop(self.target_names, axis=1).values
        self.feature_names = self.data.drop(self.target_names, axis=1).columns.values.tolist()

        _, self.n_features = self.X.shape
        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} '
                                 'exceeds n_features {}'.format(self.cat_features_idx,
                                                                self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if np.issubdtype(self.y.dtype, np.integer):
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self._REGRESSION

        self.target_values = self._get_target_values()

    def restart(self):
        """ Restarts the stream.

        It basically server the purpose of reinitializing the stream to
        its initial state.

        """
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't a tuple of (None, None) is returned.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.

        """
        self.sample_idx += batch_size
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.

        """
        return (self.n_samples - self.sample_idx) > 0

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples.

        """
        return self.n_samples - self.sample_idx

    def print_df(self):
        """
        Prints all the samples in the stream.

        """
        print(self.X)
        print(self.y)

    def get_data_info(self):
        name = self.name + ": " if self.name else ""
        if self.task_type == self._CLASSIFICATION:
            return "{}{} target(s), {} classes".format(name, self.n_targets, self.n_classes)
        elif self.task_type == self._REGRESSION:
            return "{}{} target(s)".format(name, self.n_targets)

    def _get_target_values(self):
        if self.task_type == 'classification':
            if self.n_targets == 1:
                return np.unique(self.y).tolist()
            else:
                return [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]
        elif self.task_type == self._REGRESSION:
            return [float] * self.n_targets

    def get_info(self):
        return 'DataStream(n_targets={}, target_idx={}, cat_features={}, name={})'. \
            format(self.target_idx, self.n_targets, self.cat_features,
                   self.name if not self.name else "'" + self.name + "'")


def check_data_consistency(raw_data_frame, allow_nan=False):
    """
    Check data consistency with respect to scikit-multiflow assumptions:

    * Only numeric data types are used.
    * Missing values are, in general, not supported.

    Parameters
    ----------
    raw_data_frame: pandas.DataFrame
        The data frame containing the data to check.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    """
    if (raw_data_frame.dtypes == 'object').values.any():
        # scikit-multiflow assumes that data is numeric
        raise ValueError('Non-numeric data found:\n {}'
                         'scikit-multiflow only supports numeric data.'
                         .format(raw_data_frame.dtypes))

    if raw_data_frame.isnull().values.any():
        if not allow_nan:
            raise ValueError("NaN values found. Missing values are not fully supported.\n"
                             "You can deactivate this error via the 'allow_nan' option.")
        else:
            warnings.warn("NaN values found. Functionality is not guaranteed for some methods."
                          "Proceed with caution.", UserWarning)
