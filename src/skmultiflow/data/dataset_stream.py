import pandas as pd
import numpy as np
from skmultiflow.data.base_stream import Stream


class DatasetStream(Stream):
    """ DatasetStream

    A stream generated from the entries of a dataset ( numpy array or pandas
    DataFrame).

    The stream is able to provide, as requested, a number of samples, in
    a way that old samples cannot be accessed in a later time. This is done
    so that a stream context can be correctly simulated.

    Parameters
    ----------
    raw_data:
        The dataset.
    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features_idx: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.
    """

    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

    def __init__(self, raw_data, target_idx=-1, n_targets=1, cat_features_idx=None):
        super().__init__()
        self.X = None
        self.y = None
        self.cat_features_idx = [] if cat_features_idx is None else cat_features_idx
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.task_type = None
        self.n_classes = 0
        self.datatype = type(raw_data)
        self.raw_data = raw_data

        self.__configure()

    def __configure(self):
        if not (self.datatype == type(pd.DataFrame())) and not(self.datatype == type(np.array([]))):
            raise ValueError('Raw data should be Pands DataFrame or Numpy Array, and {} object was '
                             'passed'.format(type(self.raw_data)))
        else:
            self.raw_data = pd.DataFrame(self.raw_data)

    def prepare_for_use(self):
        """ prepare_for_use

        Prepares the stream for use. This functions should always be
        called after the stream initialization.

        """
        self._load_data()
        del self.raw_data
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def _load_data(self):

        rows, cols = self.raw_data.shape
        self.n_samples = rows
        labels = self.raw_data.columns.values.tolist()

        if (self.target_idx + self.n_targets) == cols or (self.target_idx + self.n_targets) == 0:
            # Take everything to the right of target_idx
            self.y = self.raw_data.iloc[:, self.target_idx:].as_matrix()
            self.target_names = self.raw_data.iloc[:, self.target_idx:].columns.values.tolist()
        else:
            # Take only n_targets columns to the right of target_idx, use the rest as features
            self.y = self.raw_data.iloc[:, self.target_idx:self.target_idx + self.n_targets].as_matrix()
            self.target_names = labels[self.target_idx:self.target_idx + self.n_targets]

        self.X = self.raw_data.drop(self.target_names, axis=1).as_matrix()
        self.feature_names = self.raw_data.drop(self.target_names, axis=1).columns.values.tolist()

        _, self.n_features = self.X.shape
        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} '
                                 'exceeds n_features {}'.format(self.cat_features_idx, self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if self.y.dtype == np.integer:
            self.task_type = self.CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self.REGRESSION
        self.target_values = self.get_target_values()

    def restart(self):
        """ restart

        Restarts the stream's sample feeding, while keeping all of its
        parameters.

        It basically server the purpose of reinitializing the stream to
        its initial state.

        """
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

    def next_sample(self, batch_size=1):
        """ next_sample

        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't a tuple of (None, None) is returned.

        Parameters
        ----------
        batch_size: int
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
        return (self.n_samples - self.sample_idx) > 0

    def n_remaining_samples(self):
        return self.n_samples - self.sample_idx

    def print_df(self):
        print(self.X)
        print(self.y)

    def get_data_info(self):
        if self.task_type == self.CLASSIFICATION:
            return "{} target(s), {} target_values".format(self.n_targets, self.n_classes)
        elif self.task_type == self.REGRESSION:
            return "{} target(s)".format(self.n_targets)

    def get_target_values(self):
        if self.task_type == 'classification':
            if self.n_targets == 1:
                return np.unique(self.y).tolist()
            else:
                return [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]
        elif self.task_type == self.REGRESSION:
            return [float] * self.n_targets

    def get_info(self):
        return 'Dataset Stream:' + '  -  n_targets: ' + str(self.n_targets)
