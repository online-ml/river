from abc import ABCMeta, abstractmethod
from skmultiflow.core.base_object import BaseObject
from skmultiflow.data.base_stream import Stream
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
from skmultiflow.evaluation.measure_collection import WindowClassificationMeasurements, ClassificationMeasurements, \
    MultiOutputMeasurements, WindowMultiOutputMeasurements, RegressionMeasurements, WindowRegressionMeasurements


class StreamEvaluator(BaseObject, metaclass=ABCMeta):
    """ BaseEvaluator

    The abstract class that works as a base model for all of this framework's 
    evaluators. It creates a basic interface that evaluation modules should 
    follow in order to use them with all the tools available in scikit-workflow.

    This class should not me instantiated, as none of its methods, except the 
    get_class_type, are implemented.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """
    # Constants
    PERFORMANCE = 'performance'
    KAPPA = 'kappa'
    KAPPA_T = 'kappa_t'
    KAPPA_M = 'kappa_m'
    HAMMING_SCORE = 'hamming_score'
    HAMMING_LOSS = 'hamming_loss'
    EXACT_MATCH = 'exact_match'
    J_INDEX = 'j_index'
    MSE = 'mean_square_error'
    MAE = 'mean_absolute_error'
    TRUE_VS_PREDICT = 'true_vs_predicts'
    PLOT_TYPES = [PERFORMANCE,
                  KAPPA,
                  KAPPA_T,
                  KAPPA_M,
                  HAMMING_SCORE,
                  HAMMING_LOSS,
                  EXACT_MATCH,
                  J_INDEX,
                  MSE,
                  MAE,
                  TRUE_VS_PREDICT]
    CLASSIFICATION_METRICS = [PERFORMANCE,
                              KAPPA,
                              KAPPA_T,
                              KAPPA_M,
                              TRUE_VS_PREDICT]
    REGRESSION_METRICS = [MSE,
                          MAE,
                          TRUE_VS_PREDICT]
    MULTI_OUTPUT_METRICS = [HAMMING_SCORE,
                            HAMMING_LOSS,
                            EXACT_MATCH,
                            J_INDEX]
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    MULTI_OUTPUT = 'multi_output'
    SINGLE_OUTPUT = 'single-output'
    TASK_TYPES = [CLASSIFICATION,
                  REGRESSION,
                  MULTI_OUTPUT,
                  SINGLE_OUTPUT]

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

        # Metrics
        self.global_classification_metrics = None
        self.partial_classification_metrics = None

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
    def evaluate(self, stream, classifier):
        """ evaluate
        
        This function evaluates the classifier, using the class parameters, and 
        by feeding it with instances coming from the stream parameter.
        
        Parameters
        ----------
        stream: BaseInstanceStream extension
            The stream to be use in the evaluation process.
        
        classifier: BaseClassifier extension or list of BaseClassifier extensions
            The classifier or classifiers to be evaluated.
            
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifier's at the end of the evaluation process.
            
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partially fits the classifiers.
        
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        y: Array-like
            An array-like containing the class labels of all samples in X.
        
        target_values: list
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
        """ predict
        
        Predicts with the classifier, or classifiers, being evaluated.
        
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
        """ set_params
        
        Pass parameter names and values through a dictionary so that their 
        values can be updated.
        
        Parameters
        ----------
        parameter_dict: dictionary
            A dictionary where the keys are parameters' names and the values 
            are the new values for those parameters.
         
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
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
                self._output_type = self.SINGLE_OUTPUT
            elif self.stream.n_targets > 1:
                self._output_type = self.MULTI_OUTPUT
            else:
                raise ValueError('Unexpected number of outputs in stream: {}.'.format(self.stream.n_targets))
        else:
            raise ValueError('{} is not a valid stream type.'.format(self.stream))

        # Metrics configuration
        self.metrics = [x.lower() for x in self.metrics]

        for plot in self.metrics:
            if plot not in self.PLOT_TYPES:
                raise ValueError('Plot type not supported: {}.'.format(plot))

        # Check consistency between output type and metrics and between metrics
        if self._output_type == self.SINGLE_OUTPUT:
            classification_metrics = set(self.CLASSIFICATION_METRICS)
            regression_metrics = set(self.REGRESSION_METRICS)
            evaluation_metrics = set(self.metrics)

            if evaluation_metrics.union(classification_metrics) == classification_metrics:
                self._task_type = self.CLASSIFICATION
            elif evaluation_metrics.union(regression_metrics) == regression_metrics:
                self._task_type = self.REGRESSION
            else:
                raise ValueError("Inconsistent metrics {} for {} stream.".format(self.metrics, self._output_type))
        else:
            multi_output_metrics = set(self.MULTI_OUTPUT_METRICS)
            evaluation_metrics = set(self.metrics)
            if evaluation_metrics.union(multi_output_metrics) == multi_output_metrics:
                self._task_type = self.MULTI_OUTPUT
            else:
                raise ValueError("Inconsistent metrics {} for {} stream.".format(self.metrics, self._output_type))

        self._valid_configuration = True

        return self._valid_configuration

    def _init_metrics(self):
        """ _init_metrics

        Starts up the metrics and statistics watchers. One watcher is created
        for each of the learners to be evaluated.

        """
        self.global_classification_metrics = []
        self.partial_classification_metrics = []

        if self._task_type == self.CLASSIFICATION:
            for i in range(self.n_models):
                self.global_classification_metrics.append(ClassificationMeasurements())
                self.partial_classification_metrics.append(WindowClassificationMeasurements(window_size=self.n_sliding))

        elif self._task_type == self.MULTI_OUTPUT:
            for i in range(self.n_models):
                self.global_classification_metrics.append(MultiOutputMeasurements())
                self.partial_classification_metrics.append(WindowMultiOutputMeasurements(window_size=self.n_sliding))

        elif self._task_type == self.REGRESSION:
            for i in range(self.n_models):
                self.global_classification_metrics.append(RegressionMeasurements())
                self.partial_classification_metrics.append(WindowRegressionMeasurements(window_size=self.n_sliding))

    def _update_metrics(self):
        """ _update_metrics

        Updates the metrics of interest. This function creates a metrics dictionary,
        which will be sent to _update_outputs, in order to save the data (if configured)

        Creates/updates a dictionary of new evaluation points. The keys of this dictionary are
        the metrics to keep track of, and the values are two element lists, or tuples, containing
        each metric's global value and their partial value (measured from the last n_wait samples).

        If more than one learner is evaluated at once, the value from the dictionary
        will be a list of lists, or tuples, containing the global metric value and
        the partial metric value, for each of the learners.

        """
        new_points_dict = {}
        if 'performance' in self.metrics:
            new_points_dict['performance'] = [[self.global_classification_metrics[i].get_performance(),
                                               self.partial_classification_metrics[i].get_performance()]
                                              for i in range(self.n_models)]

        if 'kappa' in self.metrics:
            new_points_dict['kappa'] = [[self.global_classification_metrics[i].get_kappa(),
                                         self.partial_classification_metrics[i].get_kappa()]
                                        for i in range(self.n_models)]

        if 'kappa_t' in self.metrics:
            new_points_dict['kappa_t'] = [[self.global_classification_metrics[i].get_kappa_t(),
                                           self.partial_classification_metrics[i].get_kappa_t()]
                                          for i in range(self.n_models)]

        if 'kappa_m' in self.metrics:
            new_points_dict['kappa_m'] = [[self.global_classification_metrics[i].get_kappa_m(),
                                           self.partial_classification_metrics[i].get_kappa_m()]
                                          for i in range(self.n_models)]

        if 'hamming_score' in self.metrics:
            new_points_dict['hamming_score'] = [[self.global_classification_metrics[i].get_hamming_score(),
                                                 self.partial_classification_metrics[i].get_hamming_score()]
                                                for i in range(self.n_models)]

        if 'hamming_loss' in self.metrics:
            new_points_dict['hamming_loss'] = [[self.global_classification_metrics[i].get_hamming_loss(),
                                                self.partial_classification_metrics[i].get_hamming_loss()]
                                               for i in range(self.n_models)]

        if 'exact_match' in self.metrics:
            new_points_dict['exact_match'] = [[self.global_classification_metrics[i].get_exact_match(),
                                               self.partial_classification_metrics[i].get_exact_match()]
                                              for i in range(self.n_models)]

        if 'j_index' in self.metrics:
            new_points_dict['j_index'] = [[self.global_classification_metrics[i].get_j_index(),
                                           self.partial_classification_metrics[i].get_j_index()]
                                          for i in range(self.n_models)]

        if 'mean_square_error' in self.metrics:
            new_points_dict['mean_square_error'] = [[self.global_classification_metrics[i].get_mean_square_error(),
                                                     self.partial_classification_metrics[i].get_mean_square_error()]
                                                    for i in range(self.n_models)]

        if 'mean_absolute_error' in self.metrics:
            new_points_dict['mean_absolute_error'] = [[self.global_classification_metrics[i].get_average_error(),
                                                       self.partial_classification_metrics[i].get_average_error()]
                                                      for i in range(self.n_models)]

        if 'true_vs_predicts' in self.metrics:
            true, pred = [], []
            for i in range(self.n_models):
                t, p = self.global_classification_metrics[i].get_last()
                true.append(t)
                pred.append(p)
            new_points_dict['true_vs_predicts'] = [[true[i], pred[i]] for i in range(self.n_models)]

        shift = 0
        if self._method == 'prequential':
            shift = -self.batch_size   # Adjust index due to training after testing
        self._update_outputs(self.global_sample_count + shift, new_points_dict)

    def _update_outputs(self, current_x, new_points_dict):
        """ Update outputs of the evaluation. """
        self._update_file(current_x)
        self._update_plot(current_x, new_points_dict)

    def _init_file(self):
        # Note: 'TRUE_VS_PREDICTS' or other informative data shall not be logged into the results file since they do
        # not represent actual performance.
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
                if self.PERFORMANCE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_acc_[{}],sliding_acc_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.KAPPA in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_[{}],sliding_kappa_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.KAPPA_T in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_t_[{}],sliding_kappa_t_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.KAPPA_M in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_kappa_m_[{}],sliding_kappa_m_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.HAMMING_SCORE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_hamming_score_[{}],sliding_hamming_score_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.HAMMING_LOSS in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_hamming_loss_[{}],sliding_hamming_loss_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.EXACT_MATCH in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_exact_match_[{}],sliding_exact_match_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.J_INDEX in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_j_index_[{}],sliding_j_index_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.MSE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_mse_[{}],sliding_mse_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                if self.MAE in self.metrics:
                    for i in range(self.n_models):
                        header += ',global_mae_[{}],sliding_mae_[{}]'.\
                            format(self.model_names[i], self.model_names[i])
                f.write(header)

    def _update_file(self, current_x, ):
        if self.output_file is not None:
            # Note: Must follow order set in _init_file()
            line = str(current_x)
            if self.PERFORMANCE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_performance(),
                                                    self.partial_classification_metrics[i].get_performance())
            if self.KAPPA in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_kappa(),
                                                    self.partial_classification_metrics[i].get_kappa())
            if self.KAPPA_T in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_kappa_t(),
                                                    self.partial_classification_metrics[i].get_kappa_t())
            if self.KAPPA_M in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_kappa_m(),
                                                    self.partial_classification_metrics[i].get_kappa_m())
            if self.HAMMING_SCORE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_hamming_score(),
                                                    self.partial_classification_metrics[i].get_hamming_score())
            if self.HAMMING_LOSS in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_hamming_loss(),
                                                    self.partial_classification_metrics[i].get_hamming_loss())
            if self.EXACT_MATCH in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_exact_match(),
                                                    self.partial_classification_metrics[i].get_exact_match())
            if self.J_INDEX in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_j_index(),
                                                    self.partial_classification_metrics[i].get_j_index())
            if self.MSE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_mean_square_error(),
                                                    self.partial_classification_metrics[i].get_mean_square_error())
            if self.MAE in self.metrics:
                for i in range(self.n_models):
                    line += ',{:.6f},{:.6f}'.format(self.global_classification_metrics[i].get_average_error(),
                                                    self.partial_classification_metrics[i].get_average_error())
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

    def _update_plot(self, current_x, new_points_dict):
        """ Update evaluation plot.

        Parameters
        ----------
        current_x: int
            The current count of analysed samples.

        new_points_dict: dictionary
            A dictionary of new points, in the format described in this
            function's documentation.

        """
        if self.visualizer is not None and self.show_plot:
            self.visualizer.on_new_train_step(current_x, new_points_dict)

    def _reset_globals(self):
        self.global_sample_count = 0
