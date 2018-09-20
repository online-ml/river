from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject
from skmultiflow.data.base_stream import Stream
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
from skmultiflow.metrics import WindowClassificationMeasurements, ClassificationMeasurements, \
    MultiTargetClassificationMeasurements, WindowMultiTargetClassificationMeasurements, RegressionMeasurements, \
    WindowRegressionMeasurements, MultiTargetRegressionMeasurements, \
    WindowMultiTargetRegressionMeasurements
from skmultiflow.utils import FastBuffer
import skmultiflow.utils.constants as constants


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

        # Misc
        self._method = None
        self._task_type = None
        self._output_type = None
        self._valid_configuration = False
        self.model_names = None
        self.model = None
        self.n_models = 0
        self.stream = None

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
            multi_label_classification_metrics = set(constants.MULTI_TARGET_CLASSIFICATION_METRICS)
            multi_target_regression_metrics = set(constants.MULTI_TARGET_REGRESSION_METRICS)
            evaluation_metrics = set(self.metrics)

            if evaluation_metrics.union(multi_label_classification_metrics) == multi_label_classification_metrics:
                self._task_type = constants.MULTI_TARGET_CLASSIFICATION
            elif evaluation_metrics.union(multi_target_regression_metrics) == multi_target_regression_metrics:
                self._task_type = constants.MULTI_TARGET_REGRESSION
            else:
                raise ValueError("Inconsistent metrics {} for {} stream.".format(self.metrics, self._output_type))

        self._valid_configuration = True

        return self._valid_configuration

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
                self.current_eval_measurements.append(WindowMultiTargetClassificationMeasurements(window_size=self.n_sliding))

        elif self._task_type == constants.REGRESSION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(RegressionMeasurements())
                self.current_eval_measurements.append(WindowRegressionMeasurements(window_size=self.n_sliding))
        elif self._task_type == constants.MULTI_TARGET_REGRESSION:
            for i in range(self.n_models):
                self.mean_eval_measurements.append(MultiTargetRegressionMeasurements())
                self.current_eval_measurements.append(WindowMultiTargetRegressionMeasurements(window_size=self.n_sliding))

    def _update_metrics(self):
        """ Updates the metrics of interest. This function creates a metrics dictionary,
        which will be sent to _update_outputs, in order to save the data (if configured)

        Creates/updates a dictionary of new evaluation points. The keys of this dictionary are
        the metrics to keep track of, and the values are two element lists, or tuples, containing
        each metric's global value and their partial value (measured from the last n_wait samples).

        If more than one learner is evaluated at once, the value from the dictionary
        will be a list of lists, or tuples, containing the global metric value and
        the partial metric value, for each of the learners.

        """
        new_points_dict = {}
        if constants.ACCURACY in self.metrics:
            new_points_dict[constants.ACCURACY] = [[self.mean_eval_measurements[i].get_accuracy(),
                                                    self.current_eval_measurements[i].get_accuracy()]
                                                   for i in range(self.n_models)]

        if constants.KAPPA in self.metrics:
            new_points_dict[constants.KAPPA] = [[self.mean_eval_measurements[i].get_kappa(),
                                                 self.current_eval_measurements[i].get_kappa()]
                                                for i in range(self.n_models)]

        if constants.KAPPA_T in self.metrics:
            new_points_dict[constants.KAPPA_T] = [[self.mean_eval_measurements[i].get_kappa_t(),
                                                   self.current_eval_measurements[i].get_kappa_t()]
                                                  for i in range(self.n_models)]

        if constants.KAPPA_M in self.metrics:
            new_points_dict[constants.KAPPA_M] = [[self.mean_eval_measurements[i].get_kappa_m(),
                                                   self.current_eval_measurements[i].get_kappa_m()]
                                                  for i in range(self.n_models)]

        if constants.HAMMING_SCORE in self.metrics:
            new_points_dict[constants.HAMMING_SCORE] = [[self.mean_eval_measurements[i].get_hamming_score(),
                                                         self.current_eval_measurements[i].get_hamming_score()]
                                                        for i in range(self.n_models)]

        if constants.HAMMING_LOSS in self.metrics:
            new_points_dict[constants.HAMMING_LOSS] = [[self.mean_eval_measurements[i].get_hamming_loss(),
                                                        self.current_eval_measurements[i].get_hamming_loss()]
                                                       for i in range(self.n_models)]

        if constants.EXACT_MATCH in self.metrics:
            new_points_dict[constants.EXACT_MATCH] = [[self.mean_eval_measurements[i].get_exact_match(),
                                                       self.current_eval_measurements[i].get_exact_match()]
                                                      for i in range(self.n_models)]

        if constants.J_INDEX in self.metrics:
            new_points_dict[constants.J_INDEX] = [[self.mean_eval_measurements[i].get_j_index(),
                                                   self.current_eval_measurements[i].get_j_index()]
                                                  for i in range(self.n_models)]

        if constants.MSE in self.metrics:
            new_points_dict[constants.MSE] = [[self.mean_eval_measurements[i].get_mean_square_error(),
                                               self.current_eval_measurements[i].get_mean_square_error()]
                                              for i in range(self.n_models)]

        if constants.MAE in self.metrics:
            new_points_dict[constants.MAE] = [[self.mean_eval_measurements[i].get_average_error(),
                                               self.current_eval_measurements[i].get_average_error()]
                                              for i in range(self.n_models)]

        if constants.AMSE in self.metrics:
            new_points_dict[constants.AMSE] = [[self.mean_eval_measurements[i].get_average_mean_square_error(),
                                                self.current_eval_measurements[i].get_average_mean_square_error()]
                                               for i in range(self.n_models)]

        if constants.AMAE in self.metrics:
            new_points_dict[constants.AMAE] = [[self.mean_eval_measurements[i].get_average_absolute_error(),
                                                self.current_eval_measurements[i].get_average_absolute_error()]
                                               for i in range(self.n_models)]
        if constants.ARMSE in self.metrics:
            new_points_dict[constants.ARMSE] = [[self.mean_eval_measurements[i].get_average_root_mean_square_error(),
                                                 self.current_eval_measurements[i].get_average_root_mean_square_error()]
                                                for i in range(self.n_models)]

        if constants.TRUE_VS_PREDICTED in self.metrics:
            true, pred = [], []
            for i in range(self.n_models):
                t, p = self.mean_eval_measurements[i].get_last()
                true.append(t)
                pred.append(p)
            new_points_dict[constants.TRUE_VS_PREDICTED] = [[true[i], pred[i]] for i in range(self.n_models)]

        if constants.DATA_POINTS in self.metrics:

            targets = self.stream.target_values
            pred = []
            samples = FastBuffer(5000)

            for i in range(self.n_models):
                _, p = self.mean_eval_measurements[i].get_last()
                X = self.mean_eval_measurements[i].get_last_sample()

                pred.append(p)
                samples.add_element([X])

            new_points_dict[constants.DATA_POINTS] = [[[samples.get_queue()[i]], targets, pred[i]]
                                                      for i in range(self.n_models)]

        shift = 0
        if self._method == 'prequential':
            shift = -self.batch_size   # Adjust index due to training after testing
        self._update_outputs(self.global_sample_count + shift, new_points_dict)

    def _update_outputs(self, current_sample_id, new_points_dict):
        """ Update outputs of the evaluation. """
        self._update_file(current_sample_id)
        self._update_plot(current_sample_id, new_points_dict)

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
                if constants.ACCURACY in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_acc_[{}],sliding_acc_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.KAPPA in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_[{}],sliding_kappa_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.KAPPA_T in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_t_[{}],sliding_kappa_t_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.KAPPA_M in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_m_[{}],sliding_kappa_m_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.HAMMING_SCORE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_hamming_score_[{}],sliding_hamming_score_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.HAMMING_LOSS in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_hamming_loss_[{}],sliding_hamming_loss_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.EXACT_MATCH in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_exact_match_[{}],sliding_exact_match_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.J_INDEX in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_j_index_[{}],sliding_j_index_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.MSE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_mse_[{}],sliding_mse_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.MAE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_mae_[{}],sliding_mae_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.AMSE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_amse_[{}],sliding_amse_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.AMAE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_amae_[{}],sliding_amae_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if constants.ARMSE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_armse_[{}],sliding_armse_[{}]'.\
                            format(self.model_names[i], self.model_names[i])

                if constants.TRUE_VS_PREDICTED in self.metrics:
                    for i in range(self.n_models):
                        header += ',true_value_[{}],predicted_value_[{}]'.\
                            format(self.model_names[i], self.model_names[i])

                f.write(header)

    def _update_file(self, current_sample_id):
        if self.output_file is not None:
            # Note: Must follow order set in _init_file()
            line = str(current_sample_id)
            if constants.ACCURACY in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_accuracy(),
                                                    self.current_eval_measurements[i].get_accuracy())
            if constants.KAPPA in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_kappa(),
                                                    self.current_eval_measurements[i].get_kappa())
            if constants.KAPPA_T in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_kappa_t(),
                                                    self.current_eval_measurements[i].get_kappa_t())
            if constants.KAPPA_M in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_kappa_m(),
                                                    self.current_eval_measurements[i].get_kappa_m())
            if constants.HAMMING_SCORE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_hamming_score(),
                                                    self.current_eval_measurements[i].get_hamming_score())
            if constants.HAMMING_LOSS in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_hamming_loss(),
                                                    self.current_eval_measurements[i].get_hamming_loss())
            if constants.EXACT_MATCH in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_exact_match(),
                                                    self.current_eval_measurements[i].get_exact_match())
            if constants.J_INDEX in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_j_index(),
                                                    self.current_eval_measurements[i].get_j_index())
            if constants.MSE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_mean_square_error(),
                                                    self.current_eval_measurements[i].get_mean_square_error())
            if constants.MAE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_average_error(),
                                                    self.current_eval_measurements[i].get_average_error())
            if constants.AMSE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_average_mean_square_error(),
                                                    self.current_eval_measurements[i].get_average_mean_square_error())
            if constants.AMAE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_average_absolute_error(),
                                                    self.current_eval_measurements[i].get_average_absolute_error())
            if constants.ARMSE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.mean_eval_measurements[i].get_average_root_mean_square_error(),
                                                    self.current_eval_measurements[i].get_average_root_mean_square_error())

            if constants.TRUE_VS_PREDICTED in self.metrics:

                for i in range(self.n_models):
                    t, p = self.mean_eval_measurements[i].get_last()
                    line += ',{:.6f},{:.6f}'.format(t, p)

            with open(self.output_file, 'a') as f:
                f.write('\n' + line)

    def _init_plot(self):
        """ Initialize plot to display the evaluation results.

        """
        if self.show_plot:
            self.visualizer = EvaluationVisualizer(task_type=self._task_type,
                                                   n_sliding=self.n_sliding,
                                                   dataset_name=self.stream.get_data_info(),
                                                   plots=self.metrics,
                                                   n_learners=self.n_models,
                                                   learner_name=self.model_names)

    def _update_plot(self, current_sample_id, new_points_dict):
        """ Update evaluation plot.

        Parameters
        ----------
        current_sample_id: int
            The current count of analysed samples.

        new_points_dict: dictionary
            A dictionary of new points, in the format described in this
            function's documentation.

        """
        if self.visualizer is not None and self.show_plot:
            self.visualizer.on_new_train_step(current_sample_id, new_points_dict)

    def _reset_globals(self):
        self.global_sample_count = 0

    def evaluation_summary(self, logging, start_time, end_time):
        if end_time - start_time > self.max_time:
            logging.info('Time limit reached. Evaluation stopped.')
            logging.info('Evaluation time:     {:.2f} s'.format(self.max_time))
        else:
            logging.info('Evaluation time:     {:.2f} s'.format(end_time - start_time))
        logging.info('Processed samples: {}'.format(self.global_sample_count))
        logging.info('Mean performance:')
        for i in range(self.n_models):
            if constants.ACCURACY in self.metrics:
                logging.info('{} - Accuracy     : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_accuracy()))
            if constants.KAPPA in self.metrics:
                logging.info('{} - Kappa        : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_kappa()))
            if constants.KAPPA_T in self.metrics:
                logging.info('{} - Kappa T      : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_kappa_t()))
            if constants.KAPPA_M in self.metrics:
                logging.info('{} - Kappa M      : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_kappa_m()))
            if constants.HAMMING_SCORE in self.metrics:
                logging.info('{} - Hamming score: {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_hamming_score()))
            if constants.HAMMING_LOSS in self.metrics:
                logging.info('{} - Hamming loss : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_hamming_loss()))
            if constants.EXACT_MATCH in self.metrics:
                logging.info('{} - Exact matches: {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_exact_match()))
            if constants.J_INDEX in self.metrics:
                logging.info('{} - Jaccard index: {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_j_index()))
            if constants.MSE in self.metrics:
                logging.info('{} - MSE          : {:.4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_mean_square_error()))
            if constants.MAE in self.metrics:
                logging.info('{} - MAE          : {:4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_average_error()))
            if constants.AMSE in self.metrics:
                logging.info('{} - AMSE          : {:4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_average_mean_square_error()))
            if constants.AMAE in self.metrics:
                logging.info('{} - AMAE          : {:4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_average_absolute_error()))
            if constants.ARMSE in self.metrics:
                logging.info('{} - ARMSE          : {:4f}'.format(
                    self.model_names[i], self.mean_eval_measurements[i].get_average_root_mean_square_error()))

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
