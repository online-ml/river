import os
import logging
import warnings
from timeit import default_timer as timer
from skmultiflow.evaluation.base_evaluator import StreamEvaluator


class EvaluatePrequential(StreamEvaluator):
    """ EvaluatePrequential
    
    The prequential evaluation method, or interleaved test-then-train method, 
    is an alternative to the traditional holdout evaluation, inherited from 
    batch setting problems. 
    
    The prequential evaluation is designed specifically for stream settings, 
    in the sense that each sample serves two purposes, and that samples are 
    analysed sequentially, in order of arrival, and become immediately 
    inaccessible by the means of the stream.
    
    This method consists of using each sample to test the model, which means 
    to make a predictions or a regression, and then the same sample is used 
    to train the learner (partial fit it). This way the learner is always 
    being tested on samples that it hasn't seen yet.
    
    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
        
    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['performance', 'kappa'])
        The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
        and/or logged into the output file. Valid options are 'performance', 'kappa', 'kappa_t', 'kappa_m',
        'hamming_score', 'hamming_loss', 'exact_match', 'j_index', 'mean_square_error', 'mean_absolute_error',
        'true_vs_predicts'.
    
    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting will slow down the evaluation
        process.

    restart_stream: bool, optional (default=True)
        If True, the stream is restarted once the evaluation is complete.
    
    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    2. The metric 'true_vs_predicts' is intended to be informative only. It corresponds to evaluations at a specific
       moment which might not represent the actual learner performance across all instances. Values are not logged into
       the result file.
    
    Examples
    --------
    >>> # The first example demonstrates how to use the evaluator to evaluate one learner
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> stream = FileStream("skmultiflow/datasets/covtype.csv", -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setup the classifier
    >>> classifier = PassiveAggressiveClassifier()
    >>> # Setup the pipeline
    >>> pipe = Pipeline([('Classifier', classifier)])
    >>> # Setup the evaluator
    >>> evaluator = EvaluatePrequential(pretrain_size=200, max_samples=10000, batch_size=1, n_wait=200, max_time=1000,
    ... output_file=None, show_plot=True, metrics=['kappa', 'kappa_t', 'performance'])
    >>> # Evaluate
    >>> evaluator.evaluate(stream=stream, model=pipe)
    
    >>> # The second example will demonstrate how to compare two classifiers with
    >>> # the EvaluatePrequential
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.classification.lazy.knn_adwin import KNNAdwin
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> stream = FileStream("skmultiflow/datasets/covtype.csv", -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setup the classifiers
    >>> clf_one = PassiveAggressiveClassifier()
    >>> clf_two = KNNAdwin(k=8)
    >>> # Setup the pipeline for clf_one
    >>> pipe = Pipeline([('Classifier', clf_one)])
    >>> # Create the list to hold both classifiers
    >>> classifier = [pipe, clf_two]
    >>> # Setup the evaluator
    >>> evaluator = EvaluatePrequential(pretrain_size=200, max_samples=10000, batch_size=1, n_wait=200, max_time=1000,
    ... output_file=None, show_plot=True, metrics=['kappa', 'kappa_t', 'performance'])
    >>> # Evaluate
    >>> evaluator.evaluate(stream=stream, model=classifier)
    
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
                 restart_stream=True):

        super().__init__()
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        if metrics is None:
            self.metrics = [self.PERFORMANCE, self.KAPPA]
        else:
            self.metrics = metrics
        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ evaluate
        
        Evaluates a learner or set of learners by feeding them with the stream 
        samples.
        
        Parameters
        ----------
        stream: A stream (an extension from BaseInstanceStream) 
            The stream from which to draw the samples. 
        
        model: A learner (an extension from BaseClassifier) or a list of learners.
            The learner or learners on which to train the model and measure the 
            performance metrics.

        model_names: list, optional (Default=None)
            A list with the names of the learners.
            
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.
        
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
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        logging.info('Prequential Evaluation')
        logging.info('Evaluating %s target(s).', str(self.stream.n_targets))

        n_samples = self.stream.n_remaining_samples()
        if n_samples == -1 or n_samples > self.max_samples:
            n_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            logging.info('Pre-training on %s samples.', str(self.pretrain_size))
            X, y = self.stream.next_sample(self.pretrain_size)
            for i in range(self.n_models):
                if self._task_type != EvaluatePrequential.REGRESSION:
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                else:
                    self.model[i].partial_fit(X=X, y=y)
            self.global_sample_count += self.pretrain_size
            first_run = False
        else:
            logging.info('No pre-training.')

        update_count = 0
        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_samples) & (end_time - init_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        prediction[i].extend(self.model[i].predict(X))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.global_classification_metrics[j].add_result(y[i], prediction[j][i])
                            self.partial_classification_metrics[j].add_result(y[i], prediction[j][i])

                    self._check_progress(n_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != EvaluatePrequential.REGRESSION:
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                            else:
                                self.model[i].partial_fit(X, y)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.model[i].partial_fit(X, y)

                    if ((self.global_sample_count % self.n_wait) == 0 |
                            (self.global_sample_count >= self.max_samples) |
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        if end_time - init_time > self.max_time:
            logging.info('Time limit reached. Evaluation stopped.')
            logging.info('Evaluation time: {} s'.format(self.max_time))
        else:
            logging.info('Evaluation time: {:.3f} s'.format(end_time - init_time))
        logging.info('Total samples: {}'.format(self.global_sample_count))
        logging.info('Global performance:')
        for i in range(self.n_models):
            if 'performance' in self.metrics:
                logging.info('{} - Accuracy     : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_performance()))
            if 'kappa' in self.metrics:
                logging.info('{} - Kappa        : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa()))
            if 'kappa_t' in self.metrics:
                logging.info('{} - Kappa T      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa_t()))
            if 'kappa_m' in self.metrics:
                logging.info('{} - Kappa M      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa_m()))
            if 'hamming_score' in self.metrics:
                logging.info('{} - Hamming score: {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_hamming_score()))
            if 'hamming_loss' in self.metrics:
                logging.info('{} - Hamming loss : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_hamming_loss()))
            if 'exact_match' in self.metrics:
                logging.info('{} - Exact matches: {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_exact_match()))
            if 'j_index' in self.metrics:
                logging.info('{} - j index      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_j_index()))
            if 'mean_square_error' in self.metrics:
                logging.info('{} - MSE          : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_mean_square_error()))
            if 'mean_absolute_error' in self.metrics:
                logging.info('{} - MAE          : {:3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_average_error()))

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit

        Partially fit all the learners on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all samples in X.

        classes: list
            Stores all the target_values that may be encountered during the classification task.

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
        """ predict

        Predicts the labels of the X samples, by calling the predict 
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

    def _check_progress(self, n_samples):
        progress = self.global_sample_count - self.batch_size

        # Update progress
        if (progress % (n_samples // 20)) == 0:
            logging.info('{}%'.format(progress // (n_samples / 20) * 5))

    def set_params(self, parameter_dict):
        """ set_params

        This function allows the users to change some of the evaluator's parameters, 
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
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value

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
