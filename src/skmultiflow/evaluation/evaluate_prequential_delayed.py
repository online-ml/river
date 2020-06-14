import os
import warnings
import re
import numpy as np
from timeit import default_timer as timer

from numpy import unique

from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants
from skmultiflow.data import TimeManager


class EvaluatePrequentialDelayed(StreamEvaluator):
    """ The prequential evaluation delayed method.

    The prequential evaluation delayed is designed specifically for stream
    settings, in the sense that each sample serves two purposes, and that
    samples are analysed sequentially, in order of arrival, and are used to
    update the model only when their label are available, given their
    timestamps (arrival and available times).

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model
    (partial fit) after its label is available after a certain delay.
    This way the model is always tested on samples that it hasn't seen yet and
    updated on samples that have their labels available.

    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test. Also defines when
        to update the plot if ``show_plot=True``. Note that setting ``n_wait``
        too small can significantly slow the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the
        evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the
          metrics that will be displayed in plots and/or logged into the output
          file. Valid options are:
        | **Classification**
        | 'accuracy'
        | 'kappa'
        | 'kappa_t'
        | 'kappa_m'
        | 'true_vs_predicted'
        | 'precision'
        | 'recall'
        | 'f1'
        | 'gmean'
        | **Multi-target Classification**
        | 'hamming_score'
        | 'hamming_loss'
        | 'exact_match'
        | 'j_index'
        | **Regression**
        | 'mean_square_error'
        | 'mean_absolute_error'
        | 'true_vs_predicted'
        | **Multi-target Regression**
        | 'average_mean_squared_error'
        | 'average_mean_absolute_error'
        | 'average_root_mean_square_error'
        | **General purpose** (no plot generated)
        | 'running_time'
        | 'model_size'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation.
        Warning: Plotting can slow down the evaluation process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    data_points_for_classification: bool(Default: False)
        If True, the visualization used is a cloud of data points (only works
        for classification) and default performance metrics are ignored. If
        specific metrics are required, then they *must* be explicitly set
        using the ``metrics`` attribute.

    Notes
    -----
    1. This evaluator can process a single learner to track its performance;
       or multiple learners  at a time, to compare different models on the same
       stream.

    2. The metric 'true_vs_predicted' is intended to be informative only. It
       corresponds to evaluations at a specific moment which might not
       represent the actual learner performance across all instances.

    3. The metrics `running_time` and `model_size ` are not plotted when the
       `show_plot` option is set. Only their current value is displayed at the
       bottom of the figure. However, their values over the evaluation are
       written into the resulting csv file if the `output_file` option is set.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skmultiflow.data import TemporalDataStream
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.evaluation import EvaluatePrequentialDelayed
    >>>
    >>> # Columns used to get the data, label and time from iris_timestamp dataset
    >>> DATA_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    >>> LABEL_COLUMN = "label"
    >>> TIME_COLUMN = "timestamp"
    >>>
    >>> # Read a csv with stream data
    >>> data = pd.read_csv("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
    >>>                    "master/iris_timestamp.csv")
    >>> # Convert time column to datetime
    >>> data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
    >>> # Sort data by time
    >>> data = data.sort_values(by=TIME_COLUMN)
    >>> # Get X, y and time
    >>> X = data[DATA_COLUMNS].values
    >>> y = data[LABEL_COLUMN].values
    >>> time = data[TIME_COLUMN].values
    >>>
    >>>
    >>> # Set a delay of 1 day
    >>> delay_time = np.timedelta64(1, "D")
    >>> # Set the stream
    >>> stream = TemporalDataStream(X, y, time, sample_delay=delay_time, ordered=False)
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTreeClassifier()
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = EvaluatePrequentialDelayed(batch_size=1,
    >>>                                 pretrain_size=X.shape[0]//2,
    >>>                                 max_samples=X.shape[0],
    >>>                                 output_file='results_delay.csv',
    >>>                                 metrics=['accuracy', 'recall', 'precision', 'f1', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    >>> # The second example demonstrates how to compare two models
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skmultiflow.data import TemporalDataStream
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.bayes import NaiveBayes
    >>> from skmultiflow.evaluation import EvaluatePrequentialDelayed
    >>>
    >>> # Columns used to get the data, label and time from iris_timestamp dataset
    >>> DATA_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    >>> LABEL_COLUMN = "label"
    >>> TIME_COLUMN = "timestamp"
    >>>
    >>> # Read a csv with stream data
    >>> data = pd.read_csv("../data/datasets/iris_timestamp.csv")
    >>> # Convert time column to datetime
    >>> data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
    >>> # Sort data by time
    >>> data = data.sort_values(by=TIME_COLUMN)
    >>> # Get X, y and time
    >>> X = data[DATA_COLUMNS].values
    >>> y = data[LABEL_COLUMN].values
    >>> time = data[TIME_COLUMN].values
    >>>
    >>>
    >>> # Set a delay of 30 minutes
    >>> delay_time = np.timedelta64(30, "m")
    >>> # Set the stream
    >>> stream = TemporalDataStream(X, y, time, sample_delay=delay_time, ordered=False)
    >>>
    >>> # Set the models
    >>> ht = HoeffdingTreeClassifier()
    >>> nb = NaiveBayes()
    >>>
    >>> evaluator = EvaluatePrequentialDelayed(batch_size=1,
    >>>                                 pretrain_size=X.shape[0]//2,
    >>>                                 max_samples=X.shape[0],
    >>>                                 output_file='results_delay.csv',
    >>>                                 metrics=['accuracy', 'recall', 'precision', 'f1', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False):

        super().__init__()
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification

        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise TypeError("Attribute 'metrics' must be 'None' or 'list', passed {}".
                                     format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise TypeError("Attribute 'metrics' must be 'None' or 'list', passed {}".
                                     format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseSKMObject or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _update_classifiers(self, X, y, sample_weight):
        # check if there are samples to update
        if len(X) > 0:
            # Train
            if self.first_run:
                for i in range(self.n_models):
                    if self._task_type != constants.REGRESSION and \
                            self._task_type != constants.MULTI_TARGET_REGRESSION:
                        # Accounts for the moment of training beginning
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values,
                                                  sample_weight=sample_weight)
                        # Accounts the ending of training
                        self.running_time_measurements[i].compute_training_time_end()
                    else:
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
                        self.running_time_measurements[i].compute_training_time_end()

                    # Update total running time
                    self.running_time_measurements[i].update_time_measurements(self.batch_size)
                self.first_run = False
            else:
                for i in range(self.n_models):
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X, y, sample_weight=sample_weight)
                    self.running_time_measurements[i].compute_training_time_end()
                    self.running_time_measurements[i].update_time_measurements(self.batch_size)
            self.global_sample_count += len(X)  # self.batch_size

    def _update_metrics_delayed(self, y_true_delayed, y_pred_delayed):
        # update metrics if y_pred_delayed has items
        if len(y_pred_delayed) > 0:
            for j in range(self.n_models):
                for i in range(len(y_pred_delayed[0])):
                    self.mean_eval_measurements[j].add_result(y_true_delayed[i],
                                                              y_pred_delayed[j][i])
                    self.current_eval_measurements[j].add_result(y_true_delayed[i],
                                                                 y_pred_delayed[j][i])
            self._check_progress(self.actual_max_samples)
            if ((self.global_sample_count % self.n_wait) == 0 or
                    (self.global_sample_count >= self.max_samples) or
                    (self.global_sample_count / self.n_wait > self.update_count + 1)):
                if y_pred_delayed is not None:
                    self._update_metrics()
                self.update_count += 1

    def _transform_model_predictions(self, predictions):
        out = []
        if len(predictions) > 0:
            for j in range(predictions.shape[1]):
                out.append(predictions[:, j])
            out = np.asarray(out)
        return out

    def _transform_predictions_model(self, predictions):
        out = []
        if len(predictions) > 0:
            for j in range(predictions.shape[1]):
                l = []
                for i in range(predictions.shape[0]):
                    l.append(predictions[i,j])
                out.append(l)
        return out

    def _predict_samples(self, X):
        if X is not None:
            # Test
            prediction = [[] for _ in range(self.n_models)]
            for i in range(self.n_models):
                try:
                    # Testing time
                    self.running_time_measurements[i].compute_testing_time_begin()
                    prediction[i].extend(self.model[i].predict(X))
                    self.running_time_measurements[i].compute_testing_time_end()
                except TypeError:
                    raise TypeError("Unexpected prediction value from {}"
                                    .format(type(self.model[i]).__name__))
            # adapt prediction matrix to sample-model instead of model-sample by transposing it
            y_pred = np.asarray(prediction)
            # transform
            y_pred_T = self._transform_model_predictions(y_pred)
            return y_pred_T

    @property
    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseSKMObject extension or list of BaseClassifier extensions
            The trained classifiers.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation Delayed')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        self.actual_max_samples = self.stream.n_remaining_samples()
        if self.actual_max_samples == -1 or self.actual_max_samples > self.max_samples:
            self.actual_max_samples = self.max_samples

        self.first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            # get current batch
            X, y_true, arrival_time, available_time, sample_weight = self.stream.\
                next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y_true,
                                              classes=self.stream.target_values,
                                              sample_weight=sample_weight)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y_true,
                                              classes=unique(self.stream.target_values),
                                              sample_weight=sample_weight)
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y_true, sample_weight=sample_weight)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            self.first_run = False
            # initialize time_manager with last timestamp available
            self.time_manager = TimeManager(arrival_time[-1])

        self.update_count = 0
        print('Evaluating...')
        while ((self.global_sample_count < self.actual_max_samples) & (
                self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:

                # get current batch
                X, y_true, arrival_time, available_time, sample_weight = self.stream.\
                    next_sample(self.batch_size)

                # update current timestamp
                self.time_manager.update_timestamp(arrival_time[-1])

                # get delayed samples to update model before predicting a new batch
                X_delayed, y_true_delayed, y_pred_delayed = self.time_manager.\
                    get_available_samples()

                # transpose prediction matrix to model-sample again
                y_pred_delayed = self._transform_predictions_model(y_pred_delayed)

                self._update_metrics_delayed(y_true_delayed=y_true_delayed,
                                             y_pred_delayed=y_pred_delayed)

                # before getting new samples, update classifiers with samples
                # that are already available
                self._update_classifiers(X=X_delayed, y=y_true_delayed,
                                         sample_weight=sample_weight)

                # predict samples and get predictions
                y_pred = self._predict_samples(X)

                # add current samples to delayed queue
                self.time_manager.update_queue(X=X, y_true=y_true, y_pred=y_pred,
                                               sample_weight=sample_weight,
                                               arrival_time=arrival_time,
                                               available_time=available_time)

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # evaluate remaining samples in the delayed_queue
        # iterate over delay_queue while it has samples according to batch_size
        while self.time_manager.has_more_samples():
            # get current samples to process
            X_delayed, y_true_delayed, y_pred_delayed, sample_weight = self.time_manager.\
                next_sample(self.batch_size)
            # transpose prediction matrix to model-sample again
            y_pred_delayed = self._transform_predictions_model(y_pred_delayed)
            # update metrics
            self._update_metrics_delayed(y_true_delayed, y_pred_delayed)
            # update classifier with these samples for output models
            self._update_classifiers(X_delayed, y_true_delayed, sample_weight)

            self._end_time = timer()

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the estimators will be trained.

        y: numpy.ndarray of shape (, n_samples)
            The classification labels / target values for all samples in X.

        classes: list, optional (default=None)
            Stores all the classes that may be encountered during the
            classification task. Not used for regressors.

        sample_weight: numpy.ndarray, optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y,
                                              classes=classes,
                                              sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y,
                                              sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info
