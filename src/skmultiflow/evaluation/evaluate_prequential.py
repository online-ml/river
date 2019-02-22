import os
import warnings
from numpy import unique
from timeit import default_timer as timer
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants


class EvaluatePrequential(StreamEvaluator):
    """ The prequential evaluation method, or interleaved test-then-train method,
    is an alternative to the traditional holdout evaluation, inherited from
    batch setting problems.

    The prequential evaluation is designed specifically for stream settings,
    in the sense that each sample serves two purposes, and that samples are
    analysed sequentially, in order of arrival, and become immediately
    inaccessible.

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model
    (partial fit). This way the model is always tested on samples that it
    hasn't seen yet.

    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
        Note that setting `n_wait` too small can significantly slow the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | *Classification*
        | 'accuracy'
        | 'kappa'
        | 'kappa_t'
        | 'kappa_m'
        | 'true_vs_predicted'
        | *Multi-target Classification*
        | 'hamming_score'
        | 'hamming_loss'
        | 'exact_match'
        | 'j_index'
        | *Regression*
        | 'mean_square_error'
        | 'mean_absolute_error'
        | 'true_vs_predicted'
        | *Multi-target Regression*
        | 'average_mean_squared_error'
        | 'average_mean_absolute_error'
        | 'average_root_mean_square_error'
        | *Experimental*
        | 'running_time'
        | 'model_size'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    data_points_for_classification: bool(Default: False)
        If True , the visualization used is a cloud of data points
        (only works for classification)

    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    2. The metric 'true_vs_predicted' is intended to be informative only. It corresponds to evaluations at a specific
       moment which might not represent the actual learner performance across all instances.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    >>> # The second example demonstrates how to compare two models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.bayes import NaiveBayes
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>>
    >>> # Set the models
    >>> ht = HoeffdingTree()
    >>> nb = NaiveBayes()
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    >>> # The third example demonstrates how to evaluate one model
    >>> # and visualize the predictions using data points.
    >>> # Note: You can not in this case compare multiple models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>> # Set the evaluator
    >>> evaluator = EvaluatePrequential(max_samples=200,
    >>>                                 n_wait=1,
    >>>                                 pretrain_size=1,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy'],
    >>>                                 data_points_for_classification=True)
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

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

        if metrics is None and data_points_for_classification is False:
            self.metrics = [constants.ACCURACY, constants.KAPPA]

        elif data_points_for_classification is True:
            self.metrics = [constants.DATA_POINTS]

        else:
            self.metrics = metrics

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a learner or set of learners on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: StreamModel or list
            The learner or list of learners to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the learners.

        Returns
        -------
        StreamModel or list
            The trained learner(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
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
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                               self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        self.evaluation_summary()

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partially fit all the learners on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                self.model[i].partial_fit(X, y, classes, weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts the labels of the X samples, by calling the predict
        function of all the learners.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            A list containing the predicted labels for all instances in X in
            all learners.

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def set_params(self, parameter_dict):
        """ This function allows the users to change some of the evaluator's parameters,
        by passing a dictionary where keys are the parameters names, and values are
        the new parameters' values.

        Parameters
        ----------
        parameter_dict: Dictionary
            A dictionary where the keys are the names of attributes the user
            wants to change, and the values are the new values of those attributes.

        """
        for name, value in parameter_dict.items():
            if name == 'n_wait':
                self.n_wait = value
            elif name == 'max_samples':
                self.max_samples = value
            elif name == 'pretrain_size':
                self.pretrain_size = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'show_plot':
                self.show_plot = value

    def get_info(self):
        filename = "None"
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
        return 'Prequential Evaluator: n_wait: ' + str(self.n_wait) + \
               ' - max_samples: ' + str(self.max_samples) + \
               ' - max_time: ' + str(self.max_time) + \
               ' - output_file: ' + filename + \
               ' - batch_size: ' + str(self.batch_size) + \
               ' - pretrain_size: ' + str(self.pretrain_size) + \
               ' - task_type: ' + self._task_type + \
               ' - show_plot: ' + ('True' if self.show_plot else 'False') + \
               ' - metrics: ' + (str(self.metrics) if self.metrics is not None else 'None')
