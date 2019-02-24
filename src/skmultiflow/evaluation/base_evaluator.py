import sys
import io
from abc import ABCMeta, abstractmethod
from timeit import default_timer as timer
from skmultiflow.core.base_object import BaseObject
from skmultiflow.data.base_stream import Stream
from .evaluation_data_buffer import EvaluationDataBuffer
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
from skmultiflow.metrics import WindowClassificationMeasurements, ClassificationMeasurements, \
    MultiTargetClassificationMeasurements, WindowMultiTargetClassificationMeasurements, RegressionMeasurements, \
    WindowRegressionMeasurements, MultiTargetRegressionMeasurements, \
    WindowMultiTargetRegressionMeasurements, RunningTimeMeasurements
import skmultiflow.utils.constants as constants
from skmultiflow.utils.utils import calculate_object_size


class StreamEvaluator(BaseObject, metaclass=ABCMeta):
    """ The abstract class that works as a base model for all of this framework's
    evaluators. It creates a basic interface that evaluation modules should
    follow in order to use them with all the tools available in scikit-workflow.

    This class should not me instantiated, as none of its methods, except the
    get_class_type, are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def __init__(self):
        # Evaluator configuration
        self.n_wait = 0
        self.max_samples = 0
        self.batch_size = 0
        self.pretrain_size = 0
        self.max_time = 0
        self.metrics = []
        self.output_file = None
        self.show_plot = False
        self.restart_stream = True
        self.test_size = 0
        self.dynamic_test_set = False
        self.data_points_for_classification = False

        # Metrics
        self.mean_eval_measurements = None
        self.current_eval_measurements = None
        self._data_dict = None
        self._data_buffer = None
        self._file_buffer = ''
        self._file_buffer_size = 0

        # Misc
        self._method = None
        self._task_type = None
        self._output_type = None
        self._valid_configuration = False
        self.model_names = None
        self.model = None
        self.n_models = 0
        self.stream = None
        self._start_time = -1
        self._end_time = -1

        self.visualizer = None
        self.n_sliding = 0
        self.global_sample_count = 0

    @abstractmethod
    def evaluate(self, stream, model, model_names=None):
        """ evaluate

        Evaluates a learner or set of learners on samples from a stream.

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
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partially fits the classifiers.

        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            An array-like containing the class labels of all samples in X.

        classes: list
            A list containing all class labels of the classification problem.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifier's at the end of the evaluation process.

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """ Predicts with the classifier, or classifiers, being evaluated.

        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        Returns
        -------
        list
            A list containing the array-likes representing each classifier's
            prediction.

        """
        raise NotImplementedError

    def get_class_type(self):
        return 'evaluator'

    @abstractmethod
    def set_params(self, parameter_dict):
        """ Update parameter names and values via a dictionary.

        Parameters
        ----------
        parameter_dict: dictionary
            A dictionary where the keys are parameters' names and the values
            are the new values for those parameters.

        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        """Collects information about the evaluator.

            Returns
            -------
            string
                Evaluator description.
        """
        raise NotImplementedError

    def _init_evaluation(self, stream, model, model_names=None):
        # First, verify if this is a single evaluation or a comparison between learners.
        if isinstance(model, list):
            self.n_models = len(model)
            for m in model:
                if not hasattr(m, 'predict'):
                    raise NotImplementedError('{} does not have a predict() method.'.format(m))
        else:
            self.n_models = 1
            if not hasattr(model, 'predict'):
                raise NotImplementedError('{} does not have a predict() method.'.format(model))

        self.model = model if isinstance(model, list) else [model]
        if isinstance(stream, Stream):
            self.stream = stream
        else:
            raise ValueError('{} is not a valid stream type.'.format(stream))

        if model_names is None:
            self.model_names = ['M{}'.format(i) for i in range(self.n_models)]
        else:
            if isinstance(model_names, list):
                if len(model_names) != self.n_models:
                    raise ValueError("Number of model names does not match the number of models.")
                else:
                    self.model_names = model_names
            else:
                raise ValueError("model_names must be a list.")

    def _check_configuration(self):
        # Check stream to infer task type
        if isinstance(self.stream, Stream):
            if self.stream.n_targets == 1:
                self._output_type = constants.SINGLE_OUTPUT
            elif self.stream.n_targets > 1:
                self._output_type = constants.MULTI_OUTPUT
            else:
                raise ValueError('Unexpected number of outputs in stream: {}.'.format(self.stream.n_targets))
        else:
            raise ValueError('{} is not a valid stream type.'.format(self.stream))

        # Metrics configuration
        self.metrics = [x.lower() for x in self.metrics]

        for plot in self.metrics:
            if plot not in constants.PLOT_TYPES:
                raise ValueError('Plot type not supported: {}.'.format(plot))

        # Check consistency between output type and metrics and between metrics
        if self._output_type == constants.SINGLE_OUTPUT:
            classification_metrics = set(constants.CLASSIFICATION_METRICS)
            regression_metrics = set(constants.REGRESSION_METRICS)
            evaluation_metrics = set(self.metrics)

            if evaluation_metrics.intersection(classification_metrics) == \
                    evaluation_metrics.intersection(regression_metrics):
                self._task_type = constants.UNDEFINED
                raise ValueError("You need another metric with {}".format(self.metrics))

            elif evaluation_metrics.union(classification_metrics) == classification_metrics or \
                    self.data_points_for_classification:
                self._task_type = constants.CLASSIFICATION
            elif evaluation_metrics.union(regression_metrics) == regression_metrics:
                self._task_type = constants.REGRESSION
            else:
                raise ValueError("Inconsistent metrics {} for {} stream.".format(self.metrics, self._output_type))
        else:
            multi_target_classification_metrics = set(constants.MULTI_TARGET_CLASSIFICATION_METRICS)
            multi_target_regression_metrics = set(constants.MULTI_TARGET_REGRESSION_METRICS)
            evaluation_metrics = set(self.metrics)

            if evaluation_metrics.union(multi_target_classification_metrics) == multi_target_classification_metrics:
                self._task_type = constants.MULTI_TARGET_CLASSIFICATION
            elif evaluation_metrics.union(multi_target_regression_metrics) == multi_target_regression_metrics:
                self._task_type = constants.MULTI_TARGET_REGRESSION
            else:
                raise ValueError("Inconsistent metrics {} for {} stream.".format(self.metrics, self._output_type))

        self._valid_configuration = True

        return self._valid_configuration

    def _check_progress(self, total_samples):
        current_sample = self.global_sample_count - self.batch_size

        # Update progress
        try:
            if (current_sample % (total_samples // 20)) == 0:
                self.update_progress_bar(current_sample, total_samples, 20, timer() - self._start_time)
            if self.global_sample_count >= total_samples:
                self.update_progress_bar(current_sample, total_samples, 20, timer() - self._start_time)
                print()
        except ZeroDivisionError:
            raise ZeroDivisionError("The stream is too small to evaluate. The minimum size is 20 samples.")

    @staticmethod
    def update_progress_bar(curr, total, steps, time):
        progress = curr / total
        progress_bar = round(progress * steps)
        print('\r', '#' * progress_bar + '-' * (steps - progress_bar),
              '[{:.0%}] [{:.2f}s]'.format(progress, time), end='')
        sys.stdout.flush()    # Force flush to stdout

    def _init_metrics(self):
        """ Starts up the metrics and statistics watchers. One watcher is created
        for each of the learners to be evaluated.

        """
        self.mean_eval_measurements = []
        self.current_eval_measurements = []

        if self._task_type == constants.CLASSIFICATION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(ClassificationMeasurements())
                self.current_eval_measurements.append(WindowClassificationMeasurements(window_size=self.n_sliding))

        elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(MultiTargetClassificationMeasurements())
                self.current_eval_measurements.append(WindowMultiTargetClassificationMeasurements(
                    window_size=self.n_sliding))

        elif self._task_type == constants.REGRESSION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(RegressionMeasurements())
                self.current_eval_measurements.append(WindowRegressionMeasurements(window_size=self.n_sliding))

        elif self._task_type == constants.MULTI_TARGET_REGRESSION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(MultiTargetRegressionMeasurements())
                self.current_eval_measurements.append(WindowMultiTargetRegressionMeasurements(
                    window_size=self.n_sliding))

        # Running time
        self.running_time_measurements = []
        for i in range(self.n_models):
            self.running_time_measurements.append(RunningTimeMeasurements())

        # Evaluation data buffer
        self._data_dict = {}
        for metric in self.metrics:
            data_ids = [constants.MEAN, constants.CURRENT]
            if metric == constants.TRUE_VS_PREDICTED:
                data_ids = [constants.Y_TRUE, constants.Y_PRED]
            elif metric == constants.DATA_POINTS:
                data_ids = ['X', 'target_values', 'prediction']
            elif metric == constants.RUNNING_TIME:
                data_ids = ['training_time', 'testing_time', 'total_running_time']
            elif metric == constants.MODEL_SIZE:
                data_ids = ['model_size']
            self._data_dict[metric] = data_ids

        self._data_buffer = EvaluationDataBuffer(data_dict=self._data_dict)

    def _update_metrics(self):
        """ Updates the metrics of interest. This function updates the evaluation data buffer
        which is used to track performance during evaluation.

        The content of the buffer depends on the evaluation task type and metrics selected.

        If more than one model/learner is evaluated at once, data is stored as lists inside
        the buffer.

        """
        shift = 0
        if self._method == 'prequential':
            shift = -self.batch_size  # Adjust index due to training after testing
        sample_id = self.global_sample_count + shift

        for metric in self.metrics:
            values = [[], []]
            if metric == constants.ACCURACY:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_accuracy())
                    values[1].append(self.current_eval_measurements[i].get_accuracy())

            elif metric == constants.KAPPA:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_kappa())
                    values[1].append(self.current_eval_measurements[i].get_kappa())

            elif metric == constants.KAPPA_T:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_kappa_t())
                    values[1].append(self.current_eval_measurements[i].get_kappa_t())

            elif metric == constants.KAPPA_M:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_kappa_m())
                    values[1].append(self.current_eval_measurements[i].get_kappa_m())

            elif metric == constants.HAMMING_SCORE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_hamming_score())
                    values[1].append(self.current_eval_measurements[i].get_hamming_score())

            elif metric == constants.HAMMING_LOSS:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_hamming_loss())
                    values[1].append(self.current_eval_measurements[i].get_hamming_loss())

            elif metric == constants.EXACT_MATCH:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_exact_match())
                    values[1].append(self.current_eval_measurements[i].get_exact_match())

            elif metric == constants.J_INDEX:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_j_index())
                    values[1].append(self.current_eval_measurements[i].get_j_index())

            elif metric == constants.MSE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_mean_square_error())
                    values[1].append(self.current_eval_measurements[i].get_mean_square_error())

            elif metric == constants.MAE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_average_error())
                    values[1].append(self.current_eval_measurements[i].get_average_error())

            elif metric == constants.AMSE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_average_mean_square_error())
                    values[1].append(self.current_eval_measurements[i].get_average_mean_square_error())

            elif metric == constants.AMAE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_average_absolute_error())
                    values[1].append(self.current_eval_measurements[i].get_average_absolute_error())

            elif metric == constants.ARMSE:
                for i in range(self.n_models):
                    values[0].append(self.mean_eval_measurements[i].get_average_root_mean_square_error())
                    values[1].append(self.current_eval_measurements[i].get_average_root_mean_square_error())

            elif metric == constants.TRUE_VS_PREDICTED:
                y_true = -1
                y_pred = []
                for i in range(self.n_models):
                    t, p = self.mean_eval_measurements[i].get_last()
                    y_true = t  # We only need to keep one true value
                    y_pred.append(p)
                values[0] = y_true
                for i in range(self.n_models):
                    values[1].append(y_pred[i])

            elif metric == constants.DATA_POINTS:
                target_values = self.stream.target_values
                features = {}  # Dictionary containing feature values, using index as key

                y_pred, p = self.mean_eval_measurements[0].get_last()  # Only track one model (first) by default

                X, _ = self.stream.last_sample()
                idx_1 = 0  # TODO let the user choose the feature indices of interest
                idx_2 = 1
                features[idx_1] = X[0][idx_1]
                features[idx_2] = X[0][idx_2]

                values = [None, None, None]
                values[0] = features
                values[1] = target_values
                values[2] = y_pred

            elif metric == constants.RUNNING_TIME:
                values = [[], [], []]
                for i in range(self.n_models):
                    values[0].append(self.running_time_measurements[i].get_current_training_time())
                    values[1].append(self.running_time_measurements[i].get_current_testing_time())
                    values[2].append(self.running_time_measurements[i].get_current_total_running_time())

            elif metric == constants.MODEL_SIZE:
                values = []
                for i in range(self.n_models):
                    values.append(calculate_object_size(self.model[i], 'kB'))

            else:
                raise ValueError('Unknown metric {}'.format(metric))

            # Update buffer
            if metric == constants.TRUE_VS_PREDICTED:
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id=constants.Y_TRUE,
                                              value=values[0])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id=constants.Y_PRED,
                                              value=values[1])
            elif metric == constants.DATA_POINTS:
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='X',
                                              value=values[0])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='target_values',
                                              value=values[1])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='predictions',
                                              value=values[2])
            elif metric == constants.RUNNING_TIME:
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='training_time',
                                              value=values[0])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='testing_time',
                                              value=values[1])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='total_running_time',
                                              value=values[2])
            elif metric == constants.MODEL_SIZE:
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id='model_size',
                                              value=values)
            else:
                # Default case, 'mean' and 'current' performance
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id=constants.MEAN,
                                              value=values[0])
                self._data_buffer.update_data(sample_id=sample_id, metric_id=metric, data_id=constants.CURRENT,
                                              value=values[1])

        shift = 0
        if self._method == 'prequential':
            shift = -self.batch_size   # Adjust index due to training after testing
        self._update_outputs(self.global_sample_count + shift)

    def _update_outputs(self, sample_id):
        """ Update outputs of the evaluation. """
        self._update_file()
        if self.visualizer is not None and self.show_plot:
            self.visualizer.on_new_train_step(sample_id, self._data_buffer)

    def _init_file(self):
        if self.output_file is not None:
            with open(self.output_file, 'w+') as f:
                f.write("# TEST CONFIGURATION BEGIN")
                if hasattr(self.stream, 'get_info'):
                    f.write("\n# {}".format(self.stream.get_info()))
                for i in range(self.n_models):
                    if hasattr(self.model[i], 'get_info'):
                        f.write("\n# [{}] {}".format(self.model_names[i], self.model[i].get_info()))
                f.write("\n# {}".format(self.get_info()))
                f.write("\n# TEST CONFIGURATION END")
                header = '\nid'
                for metric in self.metrics:
                    if metric == constants.ACCURACY:
                        for i in range(self.n_models):
                            header += ',mean_acc_[{0}],current_acc_[{0}]'.format(self.model_names[i])
                    elif metric == constants.MSE:
                        for i in range(self.n_models):
                            header += ',mean_mse_[{0}],current_mse_[{0}]'.format(self.model_names[i])
                    elif metric == constants.MAE:
                        for i in range(self.n_models):
                            header += ',mean_mae_[{0}],current_mae_[{0}]'.format(self.model_names[i])
                    elif metric == constants.AMSE:
                        for i in range(self.n_models):
                            header += ',mean_amse_[{0}],current_amse_[{0}]'.format(self.model_names[i])
                    elif metric == constants.AMAE:
                        for i in range(self.n_models):
                            header += ',mean_amae_[{0}],current_amae_[{0}]'.format(self.model_names[i])
                    elif metric == constants.ARMSE:
                        for i in range(self.n_models):
                            header += ',mean_armse_[{0}],current_armse_[{0}]'.format(self.model_names[i])
                    elif metric == constants.TRUE_VS_PREDICTED:
                        header += ',true_value'
                        for i in range(self.n_models):
                            header += ',predicted_value_[{0}]'.format(self.model_names[i])
                    elif metric == constants.RUNNING_TIME:
                        for i in range(self.n_models):
                            header += ',training_time_[{0}],testing_time_[{0}],total_running_time_[{0}]'.\
                                format(self.model_names[i])
                    elif metric == constants.MODEL_SIZE:
                        for i in range(self.n_models):
                            header += ',model_size_[{0}]'.format(self.model_names[i])
                    else:
                        for i in range(self.n_models):
                            header += ',mean_{0}_[{1}],current_{0}_[{1}]'.format(metric, self.model_names[i])
                f.write(header)

    def _update_file(self):
        if self.output_file is not None:
            # Note: Must follow order set in _init_file()
            line = str(self._data_buffer.sample_id)
            for metric in self.metrics:
                if metric == constants.TRUE_VS_PREDICTED:
                    true_value = self._data_buffer.get_data(metric_id=metric, data_id=constants.Y_TRUE)
                    pred_values = self._data_buffer.get_data(metric_id=metric, data_id=constants.Y_PRED)
                    line += ',{:.6f}'.format(true_value)
                    for i in range(self.n_models):
                        line += ',{:.6f}'.format(pred_values[i])
                elif metric == constants.RUNNING_TIME:
                    training_time_values = self._data_buffer.get_data(metric_id=metric,
                                                                      data_id='training_time')
                    testing_time_values = self._data_buffer.get_data(metric_id=metric,
                                                                     data_id='testing_time')
                    total_running_time_values = self._data_buffer.get_data(metric_id=metric,
                                                                           data_id='total_running_time')
                    values = (training_time_values, testing_time_values, total_running_time_values)
                    for i in range(self.n_models):
                        line += ',{:.6f},{:.6f},{:.6f}'.format(values[0][i], values[1][i], values[2][i])
                elif metric == constants.MODEL_SIZE:
                    values = self._data_buffer.get_data(metric_id=metric, data_id='model_size')
                    for i in range(self.n_models):
                        line += ',{:.6f}'.format(values[i])
                else:
                    mean_values = self._data_buffer.get_data(metric_id=metric, data_id=constants.MEAN)
                    current_values = self._data_buffer.get_data(metric_id=metric, data_id=constants.CURRENT)
                    values = (mean_values, current_values)
                    for i in range(self.n_models):
                        line += ',{:.6f},{:.6f}'.format(values[0][i], values[1][i])

            line = '\n' + line
            if sys.getsizeof(line) + self._file_buffer_size > io.DEFAULT_BUFFER_SIZE:
                # Appending the next line will make the buffer to exceed the system's default buffer size
                # flush the content of the buffer
                self._flush_file_buffer()
            self._file_buffer += line
            self._file_buffer_size += sys.getsizeof(line)

    def _flush_file_buffer(self):
        if self._file_buffer_size > 0 and self.output_file is not None:
            with open(self.output_file, 'a') as f:
                f.write(self._file_buffer)
            self._file_buffer = ''
            self._file_buffer_size = 0

    def _init_plot(self):
        """ Initialize plot to display the evaluation results.

        """
        if self.show_plot:
            self.visualizer = EvaluationVisualizer(task_type=self._task_type,
                                                   n_wait=self.n_sliding,
                                                   dataset_name=self.stream.get_data_info(),
                                                   metrics=self.metrics,
                                                   n_models=self.n_models,
                                                   model_names=self.model_names,
                                                   data_dict=self._data_dict)

    def _reset_globals(self):
        self.global_sample_count = 0

    def evaluation_summary(self):
        if self._end_time - self._start_time > self.max_time:
            print('\nTime limit reached ({:.2f}s). Evaluation stopped.'.format(self.max_time))
        print('Processed samples: {}'.format(self.global_sample_count))
        print('Mean performance:')
        for i in range(self.n_models):
            if constants.ACCURACY in self.metrics:
                print('{} - Accuracy     : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.ACCURACY, data_id=constants.MEAN)[i]))
            if constants.KAPPA in self.metrics:
                print('{} - Kappa        : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.KAPPA, data_id=constants.MEAN)[i]))
            if constants.KAPPA_T in self.metrics:
                print('{} - Kappa T      : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.KAPPA_T, data_id=constants.MEAN)[i]))
            if constants.KAPPA_M in self.metrics:
                print('{} - Kappa M      : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.KAPPA_M, data_id=constants.MEAN)[i]))
            if constants.HAMMING_SCORE in self.metrics:
                print('{} - Hamming score: {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.HAMMING_SCORE, data_id=constants.MEAN)[i]))
            if constants.HAMMING_LOSS in self.metrics:
                print('{} - Hamming loss : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.HAMMING_LOSS, data_id=constants.MEAN)[i]))
            if constants.EXACT_MATCH in self.metrics:
                print('{} - Exact matches: {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.EXACT_MATCH, data_id=constants.MEAN)[i]))
            if constants.J_INDEX in self.metrics:
                print('{} - Jaccard index: {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.J_INDEX, data_id=constants.MEAN)[i]))
            if constants.MSE in self.metrics:
                print('{} - MSE          : {:.4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.MSE, data_id=constants.MEAN)[i]))
            if constants.MAE in self.metrics:
                print('{} - MAE          : {:4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.MAE, data_id=constants.MEAN)[i]))
            if constants.AMSE in self.metrics:
                print('{} - AMSE          : {:4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.AMSE, data_id=constants.MEAN)[i]))
            if constants.AMAE in self.metrics:
                print('{} - AMAE          : {:4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.AMAE, data_id=constants.MEAN)[i]))
            if constants.ARMSE in self.metrics:
                print('{} - ARMSE          : {:4f}'.format(
                    self.model_names[i],
                    self._data_buffer.get_data(metric_id=constants.ARMSE, data_id=constants.MEAN)[i]))
            if constants.RUNNING_TIME in self.metrics:
                # Running time
                print('{} - Training time (s)  : {:.2f}'.format(
                    self.model_names[i], self._data_buffer.get_data(metric_id=constants.RUNNING_TIME,
                                                                    data_id='training_time')[i]))
                print('{} - Testing time  (s)  : {:.2f}'.format(
                    self.model_names[i], self._data_buffer.get_data(metric_id=constants.RUNNING_TIME,
                                                                    data_id='testing_time')[i]))
                print('{} - Total time    (s)  : {:.2f}'.format(
                    self.model_names[i], self._data_buffer.get_data(metric_id=constants.RUNNING_TIME,
                                                                    data_id='total_running_time')[i]))
            if constants.MODEL_SIZE in self.metrics:
                print('{} - Size (kB)          : {:.4f}'.format(
                    self.model_names[i], self._data_buffer.get_data(metric_id=constants.MODEL_SIZE,
                                                                    data_id='model_size')[i]))

    def get_measurements(self, model_idx=None):
        """ Get measurements from the evaluation.

        Parameters
        ----------
        model_idx: int, optional (Default=None)
            Indicates the index of the model as defined in `evaluate(model)`.
            If None, returns a list with the measurements for each model.

        Returns
        -------
        tuple (mean, current)
        Mean and Current measurements. If model_idx is None, each member of the tuple
         is a a list with the measurements for each model.

        Raises
        ------
        IndexError: If the index is invalid.

        """
        if model_idx is None:
            return self.mean_eval_measurements, self.current_eval_measurements
        else:
            try:
                # Check index
                _ = self.mean_eval_measurements[model_idx]
                _ = self.current_eval_measurements[model_idx]
            except IndexError:
                print('Model index {} is invalid'.format(model_idx))
                return None, None
            return self.mean_eval_measurements[model_idx], self.current_eval_measurements[model_idx]

    def get_mean_measurements(self, model_idx=None):
        """ Get mean measurements from the evaluation.

        Parameters
        ----------
        model_idx: int, optional (Default=None)
            Indicates the index of the model as defined in `evaluate(model)`.
            If None, returns a list with the measurements for each model.

        Returns
        -------
        measurements or list
        Mean measurements. If model_idx is None, returns a list with the measurements
         for each model.

        Raises
        ------
        IndexError: If the index is invalid.

        """
        measurements, _ = self.get_measurements(model_idx)
        return measurements

    def get_current_measurements(self, model_idx=None):
        """ Get current measurements from the evaluation (measured on last `n_wait` samples).

        Parameters
        ----------
        model_idx: int, optional (Default=None)
            Indicates the index of the model as defined in `evaluate(model)`.
            If None, returns a list with the measurements for each model.

        Returns
        -------
        measurements or list
        Current measurements. If model_idx is None, returns a list with the measurements
         for each model.

        Raises
        ------
        IndexError: If the index is invalid.

        """
        _, measurements = self.get_measurements(model_idx)
        return measurements
