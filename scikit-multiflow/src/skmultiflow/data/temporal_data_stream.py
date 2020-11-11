import pandas as pd
import numpy as np

import warnings

from skmultiflow.data.data_stream import DataStream
from skmultiflow.utils import add_delay_to_timestamps


class TemporalDataStream(DataStream):
    """ Create a temporal stream from a data source.

    TemporalDataStream takes the whole data set containing the ``X`` (features),
    ``time`` (timestamps) and ``Y`` (targets).

    Parameters
    ----------
    data: numpy.ndarray or pandas.DataFrame
        The features and targets or only the features if they are passed
        in the ``y`` parameter.

    time: numpy.ndarray(dtype=datetime64) or pandas.Series (Default=None)
        The timestamp column of each instance. If its a pandas.Series, it will
        be converted into a numpy.ndarray. If None, delay by number of samples
        is considered and sample_delay must be int.

    sample_weight: numpy.ndarray or pandas.Series, optional (Default=None)
        Sample weights.

    sample_delay: numpy.ndarray, pandas.Series, numpy.timedelta64 or int, optional (Default=0)
        | Options per data type used:
        | ``numpy.timedelta64``: Samples delay in time, the time-offset \
          between the event time and when the label is available, e.g., \
          numpy.timedelta64(1,"D") for a 1-day delay)
        | ``numpy.ndarray[numpy.datetime64]``: array with the timestamps when \
          each sample will be available
        | ``pandas.Series``: series with the timestamps when each sample will\
          be available
        | ``int``: the delay in number of samples.

    y: numpy.ndarray or pandas.DataFrame, optional (Default=None)
        The targets.

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    name: str, optional (default=None)
        A string to id the data.

    ordered: bool, optional (default=True)
        If True, consider that data, y, and time are already ordered by timestamp.
        Otherwise, the data is ordered based on `time` timestamps (time cannot be
        None).

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    Notes
    -----
    The stream object provides upon request a number of samples, in a way such
    that old samples cannot be accessed at a later time. This is done to
    correctly simulate the stream context.

    """

    def __init__(self,
                 data,
                 y=None,
                 time=None,
                 sample_weight=None,
                 sample_delay=0,
                 target_idx=-1,
                 n_targets=1,
                 cat_features=None,
                 name=None,
                 allow_nan=False,
                 ordered=True):
        self.current_sample_time = None
        self.current_sample_delay = None
        self.current_sample_weight = None
        # Check if data is numpy.ndarray or pandas.DataFrame
        if not isinstance(data, np.ndarray) and not isinstance(data, pd.DataFrame):
            raise TypeError("Invalid 'data' type: {}. Valid types are np.ndarray or "
                            "pd.DataFrame".format(type(data)))
        # check if time is panda.Series or a numpy.ndarray
        if isinstance(time, pd.Series):
            self.time = pd.to_datetime(time).values
        elif isinstance(time, np.ndarray):
            self.time = np.array(time, dtype="datetime64")
        elif time is None and not isinstance(sample_delay, int):
            raise TypeError("'time' is None, 'sample_delay' must be int but {} was passed".
                            format(type(sample_delay)))
        elif time is None:
            self.time = None
        else:
            raise TypeError("Invalid 'time' type: {}. Valid types are numpy.ndarray, "
                            "pandas.Series or None.".format(type(time)))
        # check if its a single delay or a delay for instance and save delay
        if isinstance(sample_delay, np.timedelta64):
            # create delays array by adding a time delay to each timestamp
            self.sample_delay = add_delay_to_timestamps(time, sample_delay)
        elif isinstance(sample_delay, pd.Series):
            # Convert argument to datetime
            self.sample_delay = pd.to_datetime(sample_delay.values).values
        elif isinstance(sample_delay, np.ndarray):
            # Create delay array with the same time delay for all samples
            self.sample_delay = np.array(sample_delay, dtype="datetime64")
        elif isinstance(sample_delay, int):
            if self.time is not None and sample_delay != 0:
                warnings.warn("'time' will not be used because 'sample_delay' is int. "
                              "Delay by number of samples is applied. If you want to use a time "
                              "delay, use np.timedelta64 for 'sample_delay'.")
            self.time = np.arange(0, data.shape[0])
            self.sample_delay = np.arange(0 + sample_delay, data.shape[0] + sample_delay)
        else:
            raise TypeError("Invalid 'sample_delay' type: {}. Valid types are: "
                            "np.ndarray(np.datetime64), pd.Series, np.timedelta64 or int".
                            format(type(sample_delay)))

        # save sample weights if available
        if sample_weight is not None:
            self.sample_weight = sample_weight
        else:
            self.sample_weight = None
        # if data is not ordered, order it by time
        if not ordered:
            if time is not None:
                # order data based on time
                data = data[np.argsort(time)]
                # order y based on time
                y = y[np.argsort(time)]
                # order sample_weight if available
                if self.sample_weight is not None:
                    self.sample_weight = self.sample_weight[np.argsort(time)]
                # if delay is not by time, order time and delay
                if not isinstance(sample_delay, int):
                    # order sample_delay, check if not single delay
                    self.sample_delay = self.sample_delay[np.argsort(time)]
                    # order time
                    self.time.sort()
            else:
                raise TypeError("'time' is None, data cannot be ordered.")
        super().__init__(data, y, target_idx, n_targets, cat_features, name, allow_nan)

    def next_sample(self, batch_size=1):
        """
        Get next sample.

        If there is enough instances to supply at least batch_size samples,
        those are returned. If there aren't a tuple of (None, None) is returned.

        Parameters
        ----------
        batch_size: int
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next ``batch_size`` instances (``sample_x``, ``sample_y``,
            ``sample_time``, ``sample_delay`` (if available), ``sample_weight``
            (if available)). For general purposes the return can be
            treated as a numpy.ndarray.

        """
        self.sample_idx += batch_size

        try:
            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_time = self.time[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_delay = self.sample_delay[self.sample_idx -
                                                          batch_size:self.sample_idx]

            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

            # check if sample_weight is available
            if self.sample_weight is not None:
                self.current_sample_weight = self.sample_weight[self.sample_idx -
                                                                batch_size:self.sample_idx]
            else:
                self.current_sample_weight = None

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
            self.current_sample_time = None
            self.current_sample_delay = None
            self.current_sample_weight = None

        return self.current_sample_x, self.current_sample_y, self.current_sample_time,\
            self.current_sample_delay, self.current_sample_weight

    def last_sample(self):
        """ Retrieves last `batch_size` samples in the stream.

        Returns
        -------
        tuple or tuple list
            A numpy.ndarray of shape (batch_size, n_features) and an array-like of shape
            (batch_size, n_targets), representing the next batch_size samples.

        """
        return self.current_sample_x, self.current_sample_y, self.current_sample_time,\
            self.current_sample_delay, self.current_sample_weight
