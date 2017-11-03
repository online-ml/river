__author__ = 'Guilherme Matsumoto'

import logging
import warnings
from skmultiflow.evaluation.base_evaluator import BaseEvaluator
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
from skmultiflow.core.utils.utils import dict_to_tuple_list
from skmultiflow.evaluation.measure_collection import WindowClassificationMeasurements, ClassificationMeasurements, \
    MultiOutputMeasurements, WindowMultiOutputMeasurements, RegressionMeasurements, WindowRegressionMeasurements
from timeit import default_timer as timer


class EvaluatePrequential(BaseEvaluator):
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
    n_wait: int (Default: 10000)
        The number of samples to process between each holdout set test.
        Also defines when to plot points if the plot is active.
        
    max_instances: int (Default: 100000)
        The maximum number of samples to process during the evaluation.
    
    max_time: float (Default: float("inf"))
        The maximum duration of the simulation.
    
    output_file: string, optional (Default: None)
        If specified, this string defines the name of the output file. If 
        the file doesn't exist it will be created.
    
    batch_size: int (Default: 1)
        The number of samples to process at each iteration of the algorithm. 
        
    pretrain_size: int (Default: 200)
        The number of samples to use as an initial training set, which will 
        not be accounted by evaluation metrics.
    
    task_type: string (Default: 'classification')
        The type of task to execute. Can be one of the following: 'classification', 
        'regression' or 'multi_output'.
    
    show_plot: bool (Default: False)
        Whether to plot the metrics or not. Plotting will slow down the evaluation 
        process.
    
    plot_options: list, optional (Default: None)
        Which metrics to compute, and if show_plot is True, which metrics to 
        display. Plot options can contain how many of the following as the user 
        wants: 'performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 
        'exact_match', 'j_index', 'mean_square_error', 'mean_absolute_error', 
        'true_vs_predicts', 'kappa_t', 'kappa_m']
        
    Raises
    ------
    ValueError: A ValueError is raised in 2 situations. If the task type passed to 
    __init__ is not supported. Or if any of the plot options passed to __init__ is 
    not supported.
    
    Notes
    -----
    This evaluator accepts to types of evaluation processes. It can either evaluate 
    a single learner while computing its metrics or it can evaluate multiple learners 
    at a time, as a means of comparing different approaches to the same problem.
    
    Parameter show_scatter_points should only be used for small datasets, and 
    non-intensive evaluations, as it will drastically slower the evaluation process.
    
    Examples
    --------
    >>> # The first example demonstrates how to use the evaluator to evaluate one learner
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.options.file_option import FileOption
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    >>> stream = FileStream(opt, -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setup the classifier
    >>> classifier = PassiveAggressiveClassifier()
    >>> # Setup the pipeline
    >>> pipe = Pipeline([('Classifier', classifier)])
    >>> # Setup the evaluator
    >>> eval = EvaluatePrequential(pretrain_size=200, max_instances=10000, batch_size=1, n_wait=200, max_time=1000, 
    ... output_file=None, task_type='classification', show_plot=True, plot_options=['kappa', 'kappa_t', 'performance'])
    >>> # Evaluate
    >>> eval.eval(stream=stream, classifier=pipe)
    
    >>> # The second example will demonstrate how to compare two classifiers with
    >>> # the EvaluatePrequential
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.classification.lazy.knn_adwin import KNNAdwin
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.options.file_option import FileOption
    >>> from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
    >>> # Setup the File Stream
    >>> opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    >>> stream = FileStream(opt, -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setup the classifiers
    >>> clf_one = PassiveAggressiveClassifier()
    >>> clf_two = KNNAdwin(k=8)
    >>> # Setup the pipeline for clf_one
    >>> pipe = Pipeline([('Classifier', clf_one)])
    >>> # Create the list to hold both classifiers
    >>> classifier = [pipe, clf_two]
    >>> # Setup the evaluator
    >>> eval = EvaluatePrequential(pretrain_size=200, max_instances=10000, batch_size=1, n_wait=200, max_time=1000, 
    ... output_file=None, task_type='classification', show_plot=True, plot_options=['kappa', 'kappa_t', 'performance'])
    >>> # Evaluate
    >>> eval.eval(stream=stream, classifier=classifier)
    
    """

    def __init__(self, n_wait=200, max_instances=100000, max_time=float("inf"), output_file=None,
                 batch_size=1, pretrain_size=200, task_type='classification', show_plot=False, plot_options=None):

        PLOT_TYPES = ['performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 'exact_match', 'j_index',
                      'mean_square_error', 'mean_absolute_error', 'true_vs_predicts', 'kappa_t', 'kappa_m']
        TASK_TYPES = ['classification', 'regression', 'multi_output']

        super().__init__()
        self.n_wait = n_wait
        self.max_instances = max_instances
        self.max_time = max_time
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.classifier = None
        self.stream = None
        self.output_file = output_file
        self.visualizer = None
        self.n_classifiers = 0

        #plotting configs
        self.task_type = task_type.lower()
        if self.task_type not in TASK_TYPES:
            raise ValueError('Task type not supported.')
        self.show_plot = show_plot
        self.plot_options = None
        if plot_options is None:
            if self.task_type == 'classification':
                self.plot_options = ['performance', 'kappa']
            elif self.task_type == 'regression':
                self.plot_options = ['mean_square_error', 'true_vs_predict']
            elif self.task_type == 'multi_output':
                self.plot_options = ['hamming_score', 'exact_match', 'j_index']
        elif plot_options is not None:
            self.plot_options = [x.lower() for x in plot_options]

        for i in range(len(self.plot_options)):
            if self.plot_options[i] not in PLOT_TYPES:
                raise ValueError(str(self.plot_options[i]) + ': Plot type not supported.')

        #metrics
        self.global_classification_metrics = None
        self.partial_classification_metrics = None
        self.__start_metrics()

        self.global_sample_count = 0

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def eval(self, stream, classifier):
        """ eval 
        
        Evaluates a learner or set of learners by feeding them with the stream 
        samples.
        
        Parameters
        ----------
        stream: A stream (an extension from BaseInstanceStream) 
            The stream from which to draw the samples. 
        
        classifier: A learner (an extension from BaseClassifier) or a list of learners.
            The learner or learners on which to train the model and measure the 
            performance metrics.
            
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.
        
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In 
        the future, when BaseRegressor is created, it could be an axtension from that 
        class as well.
        
        """
        # First off we need to verify if this is a simple evaluation task or a comparison between learners task.
        if isinstance(classifier, type([])):
            self.n_classifiers = len(classifier)
        else:
            if hasattr(classifier, 'predict'):
                self.n_classifiers = 1
            else:
                return None

        self.__start_metrics()

        if self.show_plot:
            self.__start_plot(self.n_wait, stream.get_plot_name())

        self.__reset_globals()
        self.classifier = classifier if self.n_classifiers > 1 else [classifier]
        self.stream = stream
        self.classifier = self.__train_and_test(stream, self.classifier)

        if self.show_plot:
            self.visualizer.hold()

        return self.classifier

    def __train_and_test(self, stream=None, classifier=None):
        """ __train_and_test 
        
        Method to control the prequential evaluation, as described in the class' 
        main documentation.
        
        Parameters
        ----------
        stream: A stream (an extension from BaseInstanceStream) 
            The stream from which to draw the samples. 
        
        classifier: A learner (an extension from BaseClassifier) or a list of learners.
            The learner or learners on which to train the model and measure the 
            performance metrics.
             
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers. 
        
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In 
        the future, when BaseRegressor is created, it could be an axtension from that 
        class as well.
        
        """
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        self.classifier = classifier
        self.stream = stream
        self.__reset_globals()
        prediction = None
        logging.info('Prequential Evaluation')
        logging.info('Generating %s targets.', str(self.stream.get_num_targets()))

        rest = self.stream.estimated_remaining_instances() if (self.stream.estimated_remaining_instances() != -1 and
                                                               self.stream.estimated_remaining_instances() <=
                                                               self.max_instances) \
            else self.max_instances

        if self.output_file is not None:
            with open(self.output_file, 'w+') as f:
                f.write("# SETUP BEGIN")
                if hasattr(self.stream, 'get_info'):
                    f.write("\n# " + self.stream.get_info())
                if self.n_classifiers <= 1:
                    if hasattr(self.classifier, 'get_info'):
                        f.write("\n# " + self.classifier.get_info())
                else:
                    for i in range(self.n_classifiers):
                        if hasattr(self.classifier[i], 'get_info'):
                            f.write("\n# " + self.classifier[i].get_info())

                f.write("\n# " + self.get_info())
                f.write("\n# SETUP END")
                header = '\nx_count'
                if 'performance' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_performance_'+str(i)+',sliding_window_performance_'+str(i)
                if 'kappa' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_kappa_'+str(i)+',sliding_window_kappa_'+str(i)
                if 'kappa_t' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_kappa_t_'+str(i)+',sliding_window_kappa_t_'+str(i)
                if 'kappa_m' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_kappa_m_'+str(i)+',sliding_window_kappa_m_'+str(i)
                if 'scatter' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',true_label_'+str(i)+',prediction_'+str(i)
                if 'hamming_score' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_hamming_score_'+str(i)+',sliding_window_hamming_score_'+str(i)
                if 'hamming_loss' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_hamming_loss_'+str(i)+',sliding_window_hamming_loss_'+str(i)
                if 'exact_match' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_exact_match_'+str(i)+',sliding_window_exact_match_'+str(i)
                if 'j_index' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_j_index_'+str(i)+',sliding_window_j_index_'+str(i)
                if 'mean_square_error' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_mse_'+str(i)+',sliding_window_mse_'+str(i)
                if 'mean_absolute_error' in self.plot_options:
                    for i in range(self.n_classifiers):
                        header += ',global_mae_'+str(i)+',sliding_window_mae_'+str(i)
                f.write(header)

        first_run = True
        if (self.pretrain_size > 0):
            logging.info('Pretraining on %s samples.', str(self.pretrain_size))
            X, y = self.stream.next_instance(self.pretrain_size)
            for i in range(self.n_classifiers):
                if self.task_type != 'regression':
                    self.classifier[i].partial_fit(X=X, y=y, classes=self.stream.get_classes())
                else:
                    self.classifier[i].partial_fit(X=X, y=y)
            first_run = False

        else:
            logging.info('No pretrain.')
            """
            X, y = self.stream.next_instance()
            for i in range(self.n_classifiers):
                if self.task_type != 'regression':
                    self.classifier[i].partial_fit(X, y, self.stream.get_classes())
                else:
                    self.classifier[i].partial_fit(X, y)
            """
            first_run = False

        before_count = 0
        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_instances) & (end_time - init_time < self.max_time)
                   & (self.stream.has_more_instances())):
            try:
                X, y = self.stream.next_instance(self.batch_size)

                if X is not None and y is not None:
                    prediction = [[] for n in range(self.n_classifiers)]
                    for i in range(self.n_classifiers):
                        prediction[i].extend(self.classifier[i].predict(X))
                    self.global_sample_count += self.batch_size

                    if prediction is not None:
                        for j in range(self.n_classifiers):
                            for i in range(len(prediction[0])):
                                self.global_classification_metrics[j].add_result(y[i], prediction[j][i])
                                self.partial_classification_metrics[j].add_result(y[i], prediction[j][i])

                        nul_count = self.global_sample_count - self.batch_size

                        if ((nul_count + i + 1) % (rest / 20)) == 0:
                            logging.info('%s%%', str(((nul_count + i + 1) // (rest / 20)) * 5))

                    if first_run:
                        for i in range(self.n_classifiers):
                            if self.task_type != 'regression':
                                self.classifier[i].partial_fit(X, y, self.stream.get_classes())
                            else:
                                self.classifier[i].partial_fit(X, y)
                        first_run = False
                    else:
                        for i in range(self.n_classifiers):
                            self.classifier[i].partial_fit(X, y)

                    if ((self.global_sample_count % self.n_wait) == 0 | (
                        self.global_sample_count >= self.max_instances) |
                        (self.global_sample_count / self.n_wait > before_count + 1)):
                        before_count += 1
                        if prediction is not None:
                            self._update_metrics()

                end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    if self.show_scatter_points:
                        self._update_metrics()
                    else:
                        self._update_metrics()
                break

        if (end_time - init_time > self.max_time):
            logging.info('\nTime limit reached. Evaluation stopped.')
            logging.info('Evaluation time: %s s', str(self.max_time))
        else:
            logging.info('\nEvaluation time: %s s', str(round(end_time - init_time, 3)))
        logging.info('Total instances: %s', str(self.global_sample_count))
        for i in range(self.n_classifiers):
            if 'performance' in self.plot_options:
                logging.info('Classifier %s - Global accuracy: %s', str(i), str(round(self.global_classification_metrics[i].get_performance(), 3)))
            if 'kappa' in self.plot_options:
                logging.info('Classifier %s - Global kappa: %s', str(i), str(round(self.global_classification_metrics[i].get_kappa(), 3)))
            if 'kappa_t' in self.plot_options:
                logging.info('Classifier %s - Global kappa T: %s', str(i), str(round(self.global_classification_metrics[i].get_kappa_t(), 3)))
            if 'kappa_m' in self.plot_options:
                logging.info('Classifier %s - Global kappa M: %s', str(i), str(round(self.global_classification_metrics[i].get_kappa_m(), 3)))
            if 'scatter' in self.plot_options:
                pass
            if 'hamming_score' in self.plot_options:
                logging.info('Classifier %s - Global hamming score: %s', str(i), str(round(self.global_classification_metrics[i].get_hamming_score(), 3)))
            if 'hamming_loss' in self.plot_options:
                logging.info('Classifier %s - Global hamming loss: %s', str(i), str(round(self.global_classification_metrics[i].get_hamming_loss(), 3)))
            if 'exact_match' in self.plot_options:
                logging.info('Classifier %s - Global exact matches: %s', str(i), str(round(self.global_classification_metrics[i].get_exact_match(), 3)))
            if 'j_index' in self.plot_options:
                logging.info('Classifier %s - Global j index: %s', str(i), str(round(self.global_classification_metrics[i].get_j_index(), 3)))
            if 'mean_square_error' in self.plot_options:
                logging.info('Classifier %s - Global MSE: %s', str(i), str(round(self.global_classification_metrics[i].get_mean_square_error(), 6)))
            if 'mean_absolute_error' in self.plot_options:
                logging.info('Classifier %s - Global MAE: %s', str(i), str(round(self.global_classification_metrics[i].get_average_error(), 6)))
            if 'true_vs_predicts' in self.plot_options:
                pass

        return self.classifier

    def partial_fit(self, X, y, classes=None):
        """ partial_fit

        Partially fit all the learners on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all 
            samples in X.

        classes: list
            Stores all the classes that may be encountered during the 
            classification task.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.classifier is not None:
            for i in range(self.n_classifiers):
                self.classifier[i].partial_fit(X, y, classes)
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
        if self.classifier is not None:
            predictions = []
            for i in range(self.n_classifiers):
                predictions.append(self.classifier[i].predict(X))

        return predictions

    def __update_plot(self, current_x, new_points_dict):
        """ __update_plot

        Creates a dictionary of new points to plot. The keys of this dictionary are 
        the strings in self.plot_options, which define the metrics to keep track of, 
        and the values are two element lists, or tuples, containing each metric's 
        global value and their partial value (measured from the last n_wait samples).

        If more than one learner is evaluated at once, the value from the dictionary 
        will be a list of lists, or tuples, containing the global metric value and 
        the partial metric value, for each of the metrics.

        Parameters
        ----------
        current_x: int
            The current count of analysed samples.

        new_points_dict: dictionary
            A dictionary of new points, in the format described in this 
            function's documentation.

        """
        if self.output_file is not None:
            line = str(current_x)
            if 'performance' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_performance(), 3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_performance(), 3))
            if 'kappa' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_kappa(), 3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_kappa(), 3))
            if 'kappa_t' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_kappa_t(), 3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_kappa_t(), 3))
            if 'kappa_m' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_kappa_m(), 3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_kappa_m(), 3))
            if 'scatter' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(new_points_dict['scatter'][i][0]) + ',' + str(new_points_dict['scatter'][i][1])
            if 'hamming_score' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_hamming_score() ,3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_hamming_score(), 3))
            if 'hamming_loss' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_hamming_loss() ,3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_hamming_loss(), 3))
            if 'exact_match' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_exact_match() ,3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_exact_match(), 3))
            if 'j_index' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_j_index() ,3))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_j_index(), 3))
            if 'mean_square_error' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_mean_square_error(), 6))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_mean_square_error(), 6))
            if 'mean_absolute_error' in self.plot_options:
                for i in range(self.n_classifiers):
                    line += ',' + str(round(self.global_classification_metrics[i].get_average_error(), 6))
                    line += ',' + str(round(self.partial_classification_metrics[i].get_average_error(), 6))
            with open(self.output_file, 'a') as f:
                f.write('\n' + line)

        if self.show_plot:
            self.visualizer.on_new_train_step(current_x, new_points_dict)

    def __start_metrics(self):
        """ __start_metrics

        Starts up the metrics and statistics watchers. One watcher is created 
        for each of the learners to be evaluated.

        """
        self.global_classification_metrics = []
        self.partial_classification_metrics = []

        if self.task_type in ['classification']:
            for i in range(self.n_classifiers):
                self.global_classification_metrics.append(ClassificationMeasurements())
                self.partial_classification_metrics.append(WindowClassificationMeasurements(window_size=self.n_wait))

        elif self.task_type in ['multi_output']:
            for i in range(self.n_classifiers):
                self.global_classification_metrics.append(MultiOutputMeasurements())
                self.partial_classification_metrics.append(WindowMultiOutputMeasurements(window_size=self.n_wait))

        elif self.task_type in ['regression']:
            for i in range(self.n_classifiers):
                self.global_classification_metrics.append(RegressionMeasurements())
                self.partial_classification_metrics.append(WindowRegressionMeasurements(window_size=self.n_wait))

    def _update_metrics(self):
        """ _update_metrics
         
        Updates the metrics of interest. This function creates a metrics dictionary, 
        which will be sent to __update_plot, if the plot is enabled.

        """
        new_points_dict = {}
        if 'performance' in self.plot_options:
            new_points_dict['performance'] = [[self.global_classification_metrics[i].get_performance(),
                                               self.partial_classification_metrics[i].get_performance()]
                                              for i in range(self.n_classifiers)]

        if 'kappa' in self.plot_options:
            new_points_dict['kappa'] = [[self.global_classification_metrics[i].get_kappa(),
                                         self.partial_classification_metrics[i].get_kappa()]
                                        for i in range(self.n_classifiers)]

        if 'kappa_t' in self.plot_options:
            new_points_dict['kappa_t'] = [[self.global_classification_metrics[i].get_kappa_t(),
                                           self.partial_classification_metrics[i].get_kappa_t()]
                                          for i in range(self.n_classifiers)]

        if 'kappa_m' in self.plot_options:
            new_points_dict['kappa_m'] = [[self.global_classification_metrics[i].get_kappa_m(),
                                           self.partial_classification_metrics[i].get_kappa_m()]
                                          for i in range(self.n_classifiers)]

        if 'scatter' in self.plot_options:
            true, pred = [], []
            for i in range(self.n_classifiers):
                t, p = self.global_classification_metrics[i].get_last()
                true.append(t)
                pred.append(p)
            new_points_dict['scatter'] = [[true[i], pred[i]] for i in range(self.n_classifiers)]

        if 'hamming_score' in self.plot_options:
            new_points_dict['hamming_score'] = [[self.global_classification_metrics[i].get_hamming_score(),
                                                self.partial_classification_metrics[i].get_hamming_score()]
                                                for i in range(self.n_classifiers)]

        if 'hamming_loss' in self.plot_options:
            new_points_dict['hamming_loss'] = [[self.global_classification_metrics[i].get_hamming_loss(),
                                               self.partial_classification_metrics[i].get_hamming_loss()]
                                               for i in range(self.n_classifiers)]

        if 'exact_match' in self.plot_options:
            new_points_dict['exact_match'] = [[self.global_classification_metrics[i].get_exact_match(),
                                               self.partial_classification_metrics[i].get_exact_match()]
                                              for i in range(self.n_classifiers)]

        if 'j_index' in self.plot_options:
            new_points_dict['j_index'] = [[self.global_classification_metrics[i].get_j_index(),
                                           self.partial_classification_metrics[i].get_j_index()]
                                          for i in range(self.n_classifiers)]

        if 'mean_square_error' in self.plot_options:
            new_points_dict['mean_square_error'] = [[self.global_classification_metrics[i].get_mean_square_error(),
                                                     self.partial_classification_metrics[i].get_mean_square_error()]
                                                    for i in range(self.n_classifiers)]

        if 'mean_absolute_error' in self.plot_options:
            new_points_dict['mean_absolute_error'] = [[self.global_classification_metrics[i].get_average_error(),
                                                       self.partial_classification_metrics[i].get_average_error()]
                                                      for i in range(self.n_classifiers)]

        if 'true_vs_predicts' in self.plot_options:
            true, pred = [], []
            for i in range(self.n_classifiers):
                t, p = self.global_classification_metrics[i].get_last()
                true.append(t)
                pred.append(p)
            new_points_dict['true_vs_predicts'] = [[true[i], pred[i]] for i in range(self.n_classifiers)]

        self.__update_plot(self.global_sample_count, new_points_dict)

    def __reset_globals(self):
        self.global_sample_count = 0

    def __start_plot(self, n_wait, dataset_name):
        """ __start_plot

        Parameters
        ----------
        n_wait: int 
            The number of samples to process before each holddout set test.

        dataset_name: string
            The dataset name, will be part of the plot name.

        """
        self.visualizer = EvaluationVisualizer(n_wait=n_wait, dataset_name=dataset_name,
                                               plots=self.plot_options, n_learners=self.n_classifiers)

    def set_params(self, dict):
        """ set_params

        This function allows the users to change some of the evaluator's parameters, 
        by passing a dictionary where keys are the parameters names, and values are 
        the new parameters' values.

        Parameters
        ----------
        dict: Dictionary
            A dictionary where the keys are the names of attributes the user 
            wants to change, and the values are the new values of those attributes.

        """
        params_list = dict_to_tuple_list(dict)
        for name, value in params_list:
            if name == 'n_wait':
                self.n_wait = value
            elif name == 'max_instances':
                self.max_instances = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value


    def get_info(self):
        return 'Prequential Evaluator: n_wait: ' + str(self.n_wait) + \
               ' - max_instances: ' + str(self.max_instances) + \
               ' - max_time: ' + str(self.max_time) + \
               ' - output_file: ' + (self.output_file if self.output_file is not None else 'None') + \
               ' - batch_size: ' + str(self.batch_size) + \
               ' - pretrain_size: ' + str(self.pretrain_size) + \
               ' - task_type: ' + self.task_type + \
               ' - show_plot' + ('True' if self.show_plot else 'False') + \
               ' - plot_options: ' + (str(self.plot_options) if self.plot_options is not None else 'None')
