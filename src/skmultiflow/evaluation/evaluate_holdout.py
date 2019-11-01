import os
import warnings
import re
from timeit import default_timer as timer

from numpy import unique

from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants, get_dimensions


class EvaluateHoldout(StreamEvaluator):
    """ The holdout evaluation method or periodic holdout evaluation method.

    Analyses each arriving sample by updating its statistics, without computing
    performance metrics, nor predicting labels or regression values.

    The performance evaluation happens at every n_wait analysed samples, at which
    moment the evaluator will test the learners performance on a test set, formed
    by yet unseen samples, which will be used to evaluate performance, but not to
    train the model.

    It's possible to use the same test set for every test made or to dynamically
    create test sets, so that they differ from each other. If dynamic test sets
    are enabled, we use the data stream to create test sets on the go. This process
    is more likely to generate test sets that follow the current concept, in
    comparison to static test sets.

    Thus, if concept drift is known to be present in the stream, using dynamic
    test sets is recommended. If no concept drift is expected, disabling this
    parameter will speed up the evaluation process.

    Parameters
    ----------
    n_wait: int (Default: 10000)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
        Note that setting `n_wait` too small can significantly slow the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
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
        | **Experimental**
        | 'running_time'
        | 'model_size'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (Default=True)
        If True, the stream is restarted once the evaluation is complete.

    test_size: int (Default: 5000)
        The size of the test set.

    dynamic_test_set: bool (Default: False)
        If `True`, will continuously change the test set, otherwise will use the same test set for all tests.

    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>>
    >>> # Set the evaluator
    >>> evaluator = EvaluateHoldout(max_samples=100000,
    >>>                             max_time=1000,
    >>>                             show_plot=True,
    >>>                             metrics=['accuracy', 'kappa'],
    >>>                             dynamic_test_set=True)
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
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>> nb = NaiveBayes()
    >>>
    >>> # Set the evaluator
    >>> evaluator = EvaluateHoldout(max_samples=100000,
    >>>                             max_time=1000,
    >>>                             show_plot=True,
    >>>                             metrics=['accuracy', 'kappa'],
    >>>                             dynamic_test_set=True)
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    """

    def __init__(self,
                 n_wait=10000,
                 max_samples=100000,
                 batch_size=1,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 test_size=5000,
                 dynamic_test_set=False):

        super().__init__()
        self._method = 'holdout'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        if metrics is None:
            self.metrics = [constants.ACCURACY, constants.KAPPA]
        else:
            self.metrics = metrics
        self.restart_stream = restart_stream
        # Holdout parameters
        self.dynamic_test_set = dynamic_test_set
        if test_size < 0:
            raise ValueError('test_size has to be greater than 0.')
        else:
            self.test_size = test_size
        self.n_sliding = test_size

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
        # First off we need to verify if this is a simple evaluation task or a comparison between learners task.
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._periodic_holdout()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _periodic_holdout(self):
        """ Method to control the holdout evaluation.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Holdout Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True

        if not self.dynamic_test_set:
            print('Separating {} holdout samples.'.format(self.test_size))
            self.X_test, self.y_test = self.stream.next_sample(self.test_size)
            self.global_sample_count += self.test_size

        performance_sampling_cnt = 0
        print('Evaluating...')
        while ((self.global_sample_count < self.max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
                    self.global_sample_count += self.batch_size

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type == constants.CLASSIFICATION:
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
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            # Compute running time
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    self._check_progress(actual_max_samples)   # TODO Confirm place

                    # Test on holdout set
                    if self.dynamic_test_set:
                        perform_test = self.global_sample_count >= (self.n_wait * (performance_sampling_cnt + 1)
                                                                    + (self.test_size * performance_sampling_cnt))
                    else:
                        perform_test = (self.global_sample_count - self.test_size) % self.n_wait == 0

                    if perform_test | (self.global_sample_count >= self.max_samples):

                        if self.dynamic_test_set:
                            print('Separating {} holdout samples.'.format(self.test_size))
                            self.X_test, self.y_test = self.stream.next_sample(self.test_size)
                            self.global_sample_count += get_dimensions(self.X_test)[0]

                        # Test
                        if (self.X_test is not None) and (self.y_test is not None):
                            prediction = [[] for _ in range(self.n_models)]
                            for i in range(self.n_models):
                                try:
                                    self.running_time_measurements[i].compute_testing_time_begin()
                                    prediction[i].extend(self.model[i].predict(self.X_test))
                                    self.running_time_measurements[i].compute_testing_time_end()
                                    self.running_time_measurements[i].update_time_measurements(self.test_size)
                                except TypeError:
                                    raise TypeError("Unexpected prediction value from {}"
                                                    .format(type(self.model[i]).__name__))
                            if prediction is not None:
                                for j in range(self.n_models):
                                    for i in range(len(prediction[0])):
                                        self.mean_eval_measurements[j].add_result(self.y_test[i],
                                                                                  prediction[j][i])
                                        self.current_eval_measurements[j].add_result(self.y_test[i],
                                                                                     prediction[j][i])

                                self._update_metrics()
                            performance_sampling_cnt += 1

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

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the learners on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluateHoldout
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
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