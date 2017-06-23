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
from timeit import default_timer as timer



class EvaluatePrequential(BaseEvaluator):
    def __init__(self, n_wait=200, max_instances=100000, max_time=float("inf"), output_file=None,
                 show_performance=False, batch_size=1, pretrain_size=200, show_kappa = False, track_global_kappa=False, show_scatter_points=False):
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
        super().__init__()
        # default values
        self.n_wait = n_wait
        self.max_instances = max_instances
        self.max_time = max_time
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.show_performance = show_performance
        self.show_kappa = show_kappa
        self.show_scatter_points = show_scatter_points
        self.track_global_kappa = track_global_kappa
        self.classifier = None
        self.stream = None
        self.output_file = output_file
        self.visualizer = None
        # performance stats
        self.global_correct_predicts = 0
        self.partial_correct_predicts = 0
        self.global_sample_count = 0
        self.partial_sample_count = 0
        self.global_accuracy = 0
        # kappa stats
        self.global_kappa = 0.0
        self.all_labels = []
        self.all_predicts = []
        self.kappa_count = 0
        self.kappa_predicts = FastBuffer(n_wait)
        self.kappa_true_labels = FastBuffer(n_wait)

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")

    def eval(self, stream, classifier):
        if self.show_performance or self.show_kappa or self.show_scatter_points:
            self.start_plot(self.n_wait, stream.get_plot_name())
        self.classifier = classifier
        self.stream = stream
        self.classifier = self.train_and_test(stream, classifier)
        if self.show_performance or self.show_kappa or self.show_scatter_points:
            self.visualizer.hold()
        return self.classifier

    def train_and_test(self, stream = None, classifier = None):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        self.classifier = classifier
        self.stream = stream
        self._reset_partials()
        self._reset_globals()
        prediction = None
        logging.info('Generating %s classes.', str(self.stream.get_num_classes()))

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
                f.write("\nx_count,global_performance,partial_performance,global_kappa,sliding_window_kappa,true_label,prediction")

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
                    self.partial_sample_count += self.batch_size
                    self.kappa_predicts.add_element(np.ravel(prediction))
                    self.kappa_true_labels.add_element(np.ravel(y))
                    for i in range(len(prediction)):
                        nul_count = self.global_sample_count - self.batch_size
                        if ((prediction[i] == y[i]) and not (self.global_sample_count > self.max_instances)):
                            self.partial_correct_predicts += 1
                            self.global_correct_predicts += 1
                        if ((nul_count + i + 1) % (rest/20)) == 0:
                            logging.info('%s%%', str(((nul_count+i+1) // (rest / 20)) * 5))
                        #if self.show_scatter_points:
                            #self.visualizer.on_new_scatter_data(self.global_sample_count - self.batch_size + i, y[i],
                                                                #prediction[i])
                        self.all_labels.extend(y)
                        self.all_predicts.extend(prediction)
                    self.classifier.partial_fit(X, y)

                    if ((self.global_sample_count % self.n_wait) == 0 | (self.global_sample_count >= self.max_instances) |
                        (self.global_sample_count / self.n_wait > before_count + 1)):
                        before_count += 1
                        self.kappa_count += 1
                        self.update_metrics(y[-1], prediction[-1])
                end_time = timer()
            except BaseException as exc:
                if exc is KeyboardInterrupt:
                    self.kappa_count += 1
                    if self.show_scatter_points:
                        self.update_metrics(y[-1], prediction[-1])
                    else:
                        self.update_metrics()
                break

        if (end_time-init_time > self.max_time):
            logging.info('\nTime limit reached. Evaluation stopped.')
            logging.info('Evaluation time: %s s', str(self.max_time))
        else:
            logging.info('\nEvaluation time: %s s', str(round(end_time - init_time, 3)))
        logging.info('Total instances: %s', str(self.global_sample_count))
        logging.info('Global accuracy: %s', str(round(self.global_correct_predicts/self.global_sample_count, 3)))
        if self.track_global_kappa:
            logging.info('Global kappa: %s', str(round(self.global_kappa, 3)))

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

    def update_plot(self, partial_accuracy=None, num_instances=None, y=None, prediction=None):
        if self.output_file is not None:
            line = str(num_instances)
            i = 0
            if self.show_performance:
                line += ',' + str(round(self.global_accuracy, 3))
                line += ',' + str(round(partial_accuracy[i],3))
                i += 1
            else:
                line += ',nan,nan'
            if self.show_kappa:
                if self.track_global_kappa:
                    line += ',' +str(round(self.global_kappa, 3))
                else:
                    line += ',nan'
                line += ',' + str(round(partial_accuracy[i], 3))
            else:
                line += ',nan,nan'
            if self.show_scatter_points:
                line += ',' + str(y) + ',' + str(prediction)
            else:
                line += ',nan,nan'
            with open(self.output_file, 'a') as f:
                f.write('\n'+line)
        self.visualizer.on_new_train_step(partial_accuracy, num_instances, y, prediction, self.global_kappa)
        pass

    def update_metrics(self, y=None, prediction=None):
        """ Updates the metrics of interest.
        
            It's possible that cohen_kappa_score will return a NaN value, which happens if the predictions
            and the true labels are in perfect accordance, causing pe=1, which results in a division by 0.
            If this is detected the plot will assume it to be 1.
        
        :return: No return.
        """
        self.global_accuracy = ((self.global_sample_count - self.partial_sample_count) / self.global_sample_count) * \
                               self.global_accuracy + (self.partial_sample_count / self.global_sample_count) * \
                                                      (self.partial_correct_predicts/self.partial_sample_count)
        partial_kappa = 0.0
        partial_kappa = cohen_kappa_score(self.kappa_predicts.get_queue(), self.kappa_true_labels.get_queue())
        self.global_kappa = cohen_kappa_score(self.all_labels, self.all_predicts)
        #logging.info('%s', str(round(partial_kappa, 3)))
        if math.isnan(partial_kappa):
            partial_kappa = 1.0

        partials = None
        if self.show_kappa or self.show_performance:
            partials = []
            if self.show_performance:
                partials.append(self.partial_correct_predicts/self.partial_sample_count)
            if self.show_kappa:
                partials.append(partial_kappa)

        self.update_plot(partials, self.global_sample_count, y, prediction)

        self._reset_partials()

    def _reset_partials(self):
        self.partial_sample_count = 0
        self.partial_correct_predicts = 0

    def _reset_globals(self):
        self.global_sample_count = 0
        self.global_correct_predicts = 0
        self.global_accuracy = 0.0

    def start_plot(self, n_wait, dataset_name):
        self.visualizer = EvaluationVisualizer(n_wait=n_wait, dataset_name=dataset_name,
                                               show_performance=self.show_performance, show_kappa= self.show_kappa,
                                               track_global_kappa=self.track_global_kappa,
                                               show_scatter_points=self.show_scatter_points)
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
        plot = 'True' if self.show_performance else 'False'
        kappa = 'True' if self.show_kappa else 'False'
        scatter = 'True' if self.show_scatter_points else 'False'
        return 'Prequential Evaluator: n_wait: ' + str(self.n_wait) + \
               '  -  max_instances: ' + str(self.max_instances) + \
               '  -  max_time: ' + str(self.max_time) + \
               '  -  batch_size: ' + str(self.batch_size) + \
               '  -  pretrain_size: ' + str(self.pretrain_size) + \
               '  -  show_performance: ' + plot + \
               '  -  show_kappa: ' + kappa + \
               '  -  show_scatter_points: ' + scatter
