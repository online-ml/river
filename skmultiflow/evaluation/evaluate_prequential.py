__author__ = 'Guilherme Matsumoto'

import numpy as np
import math
import logging
import warnings
import time
from skmultiflow.evaluation.base_evaluator import BaseEvaluator
from sklearn.metrics import cohen_kappa_score
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer
from skmultiflow.core.utils.utils import dict_to_tuple_list
from skmultiflow.core.utils.data_structures import FastBuffer
from skmultiflow.evaluation.measure_collection import WindowClassificationMeasurements, ClassificationMeasurements, MultiOutputMeasurements, WindowMultiOutputMeasurements
from timeit import default_timer as timer


class EvaluatePrequential(BaseEvaluator):
    def __init__(self, n_wait=200, max_instances=100000, max_time=float("inf"), output_file=None,
                 batch_size=1, pretrain_size=200, task_type='classification', show_plot=False, plot_options=None):
        """
            Parameter show_scatter_points should only be used for small datasets, and non-intensive evaluations, as it
            will drastically slower the evaluation process.

        :param n_wait: int. Number of samples processed between metric updates, including plot points
        :param max_instances: int. Maximum number of samples to be processed
        :param max_time: int. Maximum amount of time, in seconds, that the evaluation can take
        :param output_file: string. Output file name. If given this is where the evaluation log will be saved.
        :param show_plot: boolean. If true a plot including the performance evolution will be shown.
        :param batch_size: int. The size of each batch, which means, how many samples will be treated at a time.
        :param pretrain_size: int. How many samples will be used to pre train de model. These won't be considered for metrics calculation.
        :param show_kappa: boolean. If true the visualization module will display the Kappa statistic plot.
        :param track_global_kappa: If true will keep track of a global kappa statistic. Will consume more memory and will be plotted if show_kappa is True.
        :param show_scatter_points: boolean. If True the visualization module will display a scatter of True labels vs Predicts.
        """
        PLOT_TYPES = ['performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 'exact_match', 'j_index']
        TASK_TYPES = ['classification', 'regression', 'multi_output']
        super().__init__()
        # default values
        self.n_wait = n_wait
        self.max_instances = max_instances
        self.max_time = max_time
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.classifier = None
        self.stream = None
        self.output_file = output_file
        self.visualizer = None

        #plotting configs
        self.task_type = task_type.lower()
        if self.task_type not in TASK_TYPES:
            raise ValueError('Task type not supported.')
        self.show_plot = show_plot
        self.plot_options = None
        if self.show_plot is True and plot_options is None:
            if self.task_type == 'classification':
                self.plot_options = ['performance', 'kappa']
            elif self.task_type == 'regression':
                self.plot_options = ['performance', 'scatter']
            elif self.task_type == 'multi_output':
                self.plot_options = ['hamming_score', 'exact_match', 'j_index']
        elif self.show_plot is True and plot_options is not None:
            self.plot_options = [x.lower() for x in plot_options]
        for i in range(len(self.plot_options)):
            if self.plot_options[i] not in PLOT_TYPES:
                raise ValueError('Plot type not supported.')

        #metrics
        self.global_classification_metrics = None
        self.partial_classification_metrics = None
        if self.task_type in ['classification', 'regression']:
            self.global_classification_metrics = ClassificationMeasurements()
            self.partial_classification_metrics = WindowClassificationMeasurements(window_size=self.n_wait)
        elif self.task_type in ['multi_output']:
            self.global_classification_metrics = MultiOutputMeasurements()
            self.partial_classification_metrics = WindowMultiOutputMeasurements()

        self.global_sample_count = 0

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")

    def eval(self, stream, classifier):
        if self.show_plot:
            self.start_plot(self.n_wait, stream.get_plot_name())
        self.classifier = classifier
        self.stream = stream
        self.classifier = self.train_and_test(stream, classifier)
        if self.show_plot:
            self.visualizer.hold()
        return self.classifier

    def train_and_test(self, stream=None, classifier=None):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        self.classifier = classifier
        self.stream = stream
        self._reset_globals()
        prediction = None
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
                if hasattr(self.classifier, 'get_info'):
                    f.write("\n# " + self.classifier.get_info())
                f.write("\n# " + self.get_info())
                f.write("\n# SETUP END")
                header = '\nx_count'
                if 'performance' in self.plot_options:
                    header += ',global_performance,sliding_window_performance'
                if 'kappa' in self.plot_options:
                    header += ',global_kappa,sliding_window_kappa'
                if 'scatter' in self.plot_options:
                    header += ',true_label,prediction'
                if 'hamming_score' in self.plot_options:
                    header += ',global_hamming_score,sliding_window_hamming_score'
                if 'hamming_loss' in self.plot_options:
                    header += ',global_hamming_loss,sliding_window_hamming_loss'
                if 'exact_match' in self.plot_options:
                    header += ',global_exact_match,sliding_window_exact_match'
                if 'j_index' in self.plot_options:
                    header += ',global_j_index,sliding_window_j_index'
                f.write(header)

        if (self.pretrain_size > 0):
            logging.info('Pretraining on %s samples.', str(self.pretrain_size))
            X, y = self.stream.next_instance(self.pretrain_size)
            self.classifier.partial_fit(X, y, self.stream.get_classes())
        else:
            X, y = None, None

        before_count = 0
        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_instances) & (end_time - init_time < self.max_time)
                   & (self.stream.has_more_instances())):
            try:
                X, y = self.stream.next_instance(self.batch_size)
                if X is not None and y is not None:
                    prediction = self.classifier.predict(X)
                    self.global_sample_count += self.batch_size
                    for i in range(len(prediction)):
                        self.global_classification_metrics.add_result(y[i], prediction[i])
                        self.partial_classification_metrics.add_result(y[i], prediction[i])
                        nul_count = self.global_sample_count - self.batch_size
                        if ((nul_count + i + 1) % (rest / 20)) == 0:
                            logging.info('%s%%', str(((nul_count + i + 1) // (rest / 20)) * 5))
                            # if self.show_scatter_points:
                            # self.visualizer.on_new_scatter_data(self.global_sample_count - self.batch_size + i, y[i],
                            # prediction[i])
                    self.classifier.partial_fit(X, y)

                    if ((self.global_sample_count % self.n_wait) == 0 | (
                        self.global_sample_count >= self.max_instances) |
                        (self.global_sample_count / self.n_wait > before_count + 1)):
                        before_count += 1
                        self.update_metrics()
                end_time = timer()
            except BaseException as exc:
                if exc is KeyboardInterrupt:
                    if self.show_scatter_points:
                        self.update_metrics()
                    else:
                        self.update_metrics()
                break

        if (end_time - init_time > self.max_time):
            logging.info('\nTime limit reached. Evaluation stopped.')
            logging.info('Evaluation time: %s s', str(self.max_time))
        else:
            logging.info('\nEvaluation time: %s s', str(round(end_time - init_time, 3)))
        logging.info('Total instances: %s', str(self.global_sample_count))

        if 'performance' in self.plot_options:
            logging.info('Global accuracy: %s', str(round(self.global_classification_metrics.get_performance(), 3)))
        if 'kappa' in self.plot_options:
            logging.info('Global kappa: %s', str(round(self.global_classification_metrics.get_kappa(), 3)))
        if 'scatter' in self.plot_options:
            pass
        if 'hamming_score' in self.plot_options:
            logging.info('Global hamming score: %s', str(round(self.global_classification_metrics.get_hamming_score(), 3)))
        if 'hamming_loss' in self.plot_options:
            logging.info('Global hamming loss: %s', str(round(self.global_classification_metrics.get_hamming_loss(), 3)))
        if 'exact_match' in self.plot_options:
            logging.info('Global exact matches: %s', str(round(self.global_classification_metrics.get_exact_match(), 3)))
        if 'j_index' in self.plot_options:
            logging.info('Global j index: %s', str(round(self.global_classification_metrics.get_j_index(), 3)))

        return self.classifier

    def partial_fit(self, X, y):
        if self.classifier is not None:
            self.classifier.partial_fit(X, y)
            return self
        else:
            return self

    def predict(self, X):
        if self.classifier is not None:
            self.classifier.predict(X)
            return self
        else:
            return self

    def update_plot(self, current_x, new_points_dict):
        if self.output_file is not None:
            line = str(current_x)
            if 'classification' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_performance(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_performance(), 3))
            if 'kappa' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_kappa(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_kappa(), 3))
            if 'scatter' in self.plot_options:
                line += ',' + str(new_points_dict['scatter'][0]) + ',' + str(new_points_dict['scatter'][1])
            if 'hamming_score' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_hamming_score() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_hamming_score(), 3))
            if 'hamming_loss' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_hamming_loss() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_hamming_loss(), 3))
            if 'exact_match' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_exact_match() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_exact_match(), 3))
            if 'j_index' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_j_index() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_j_index(), 3))
            with open(self.output_file, 'a') as f:
                f.write('\n' + line)

        self.visualizer.on_new_train_step(current_x, new_points_dict)
        pass

    def update_metrics(self):
        """ Updates the metrics of interest.

            It's possible that cohen_kappa_score will return a NaN value, which happens if the predictions
            and the true labels are in perfect accordance, causing pe=1, which results in a division by 0.
            If this is detected the plot will assume it to be 1.

        :return: No return.
        """

        new_points_dict = {}

        if 'performance' in self.plot_options:
            new_points_dict['performance'] = [self.global_classification_metrics.get_performance(), self.partial_classification_metrics.get_performance()]
        if 'kappa' in self.plot_options:
            new_points_dict['kappa'] = [self.global_classification_metrics.get_kappa(), self.partial_classification_metrics.get_kappa()]
        if 'scatter' in self.plot_options:
            true, pred = self.global_classification_metrics.get_last()
            new_points_dict['scatter'] = [true, pred]
        if 'hamming_score' in self.plot_options:
            new_points_dict['hamming_score'] = [self.global_classification_metrics.get_hamming_score(), self.partial_classification_metrics.get_hamming_score()]
        if 'hamming_loss' in self.plot_options:
            new_points_dict['hamming_loss'] = [self.global_classification_metrics.get_hamming_loss(), self.partial_classification_metrics.get_hamming_loss()]
        if 'exact_match' in self.plot_options:
            new_points_dict['exact_match'] = [self.global_classification_metrics.get_exact_match(), self.partial_classification_metrics.get_exact_match()]
        if 'j_index' in self.plot_options:
            new_points_dict['j_index'] = [self.global_classification_metrics.get_j_index(), self.partial_classification_metrics.get_j_index()]
        self.update_plot(self.global_sample_count, new_points_dict)

    def _reset_globals(self):
        self.global_sample_count = 0

    def start_plot(self, n_wait, dataset_name):
        self.visualizer = EvaluationVisualizer(n_wait=n_wait, dataset_name=dataset_name, plots=self.plot_options)
        pass

    def set_params(self, dict):
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
            elif name == 'show_performance':
                self.show_performance = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value
            elif name == 'show_kappa':
                self.show_kappa = value
            elif name == 'show_scatter_points':
                self.show_scatter_points = value

    def get_info(self):
        plot = 'True' if self.show_plot else 'False'
        return 'Prequential Evaluator: n_wait: ' + str(self.n_wait) + \
               '  -  max_instances: ' + str(self.max_instances) + \
               '  -  max_time: ' + str(self.max_time) + \
               '  -  batch_size: ' + str(self.batch_size) + \
               '  -  pretrain_size: ' + str(self.pretrain_size) + \
               '  -  show_plot: ' + plot + \
               '  -  plot_options: ' + str(self.plot_options)
