__author__ = 'Guilherme Matsumoto'

from skmultiflow.evaluation.BaseEvaluator import BaseEvaluator
from skmultiflow.classification.Perceptron import PerceptronMask
from skmultiflow.visualization.EvaluationVisualizer import EvaluationVisualizer
from skmultiflow.core.utils.Utils import dict_to_tuple_list
import sys, argparse
from timeit import default_timer as timer
import numpy as np
import logging



class EvaluatePrequential(BaseEvaluator):

    def __init__(self, n_wait=200, max_instances=100000, max_time=float("inf"), output_file=None,
                 show_plot=False, batch_size=1, pretrain_size=200):
        super().__init__()
        # default values
        self.n_wait = n_wait
        self.max_instances = max_instances
        self.max_time = max_time
        self.classifier = None
        self.stream = None
        self.output_file = output_file
        self.visualizer = None
        self.global_correct_predicts = 0
        self.partial_correct_predicts = 0
        self.global_sample_count = 0
        self.partial_sample_count = 0
        self.global_accuracy = 0
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.show_plot = show_plot
        pass

    # Most likely this function won't be used. I'll build an external parser later
    def parse(self, argv):
        parser = argparse.ArgumentParser(description='Testing argparse module')

        parser.add_argument("-l", dest='classifier', type=str, help='Classifier to train', default='NaiveBayes')
        parser.add_argument("-s", dest='stream', type=str, help='Stream to train', default='RandomTree')
        parser.add_argument("-e", dest='performance', type=str, help='Classification performance evaluation method')
        parser.add_argument("-i", dest='maxInt', type=int, help='Maximum number of instances')
        parser.add_argument("-t", dest='max_time', type=int, help='Max number of seconds')
        parser.add_argument("-f", dest='n_wait', type=int,
                            help='How many instances between samples of the learning performance')
        parser.add_argument("-b", dest='maxSize', type=int, help='Maximum size of model')
        parser.add_argument("-O", dest='out', type=str, help='Output file')

        args = parser.parse_args()

        print(args)

        if (args.classifier is not None):
            split = args.classifier.split()
            if len(split) > 1:
                args.classifier = split

        if args.stream is not None:
            split = args.stream.split()
            if len(split) > 1:
                args.stream = split

        return args

    def eval(self, stream, classifier):
        if self.show_plot:
            self.start_plot(self.n_wait, stream.get_plot_name())
        self.classifier = classifier
        self.stream = stream
        self.classifier = self.train_and_test(stream, classifier)
        return self.classifier

    def train_and_test(self, stream = None, classifier = None):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        self.classifier = classifier
        self.stream = stream
        end_time = timer()
        self._reset_partials()
        self._reset_globals()
        prediction = None
        logging.info('Generating %s classes.', str(self.stream.get_num_classes()))

        if (self.pretrain_size > 1):
            msg = 'Pretraining on ' + str(self.pretrain_size) + ' samples.'
            logging.info('Pretraining on %s samples.', str(self.pretrain_size))
            X, y = self.stream.next_instance(self.pretrain_size)
            #self.classifier.partial_fit(X, y, self.stream.get_classes(), True)
            self.classifier.partial_fit(X, y, self.stream.get_classes())
        else:
            X, y = None, None

        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_instances) & (timer() - init_time < self.max_time)
                   & (self.stream.has_more_instances())):
            X, y = self.stream.next_instance(self.batch_size)
            if X is not None and y is not None:
                prediction = self.classifier.predict(X)
                self.global_sample_count += self.batch_size
                self.partial_sample_count += self.batch_size
                for i in range(len(prediction)):
                    nul_count = self.global_sample_count - self.batch_size
                    if ((prediction[i] == y[i]) and not (self.global_sample_count > self.max_instances)):
                        self.partial_correct_predicts += 1
                        self.global_correct_predicts += 1
                    rest = self.stream.estimated_remaining_instances() if (self.stream.estimated_remaining_instances() != -1 and
                                                                      self.stream.estimated_remaining_instances() <=
                                                                      self.max_instances) \
                        else self.max_instances
                    if ((nul_count + i + 1) % (rest/20)) == 0:
                        logging.info('%s%%', str(((nul_count+i+1) // (rest / 20)) * 5))
                self.classifier.partial_fit(X, y)

                if ((self.global_sample_count % self.n_wait) == 0 | (self.global_sample_count >= self.max_instances)):
                    self.update_metrics()

        end_time = timer()
        logging.info('Evaluation time: %s', str(round(end_time - init_time, 3)))
        logging.info('Global accuracy: %s', str(round(self.global_correct_predicts/self.global_sample_count, 3)))
        logging.info('Total instances: %s', str(self.global_sample_count))
        self.visualizer.hold()
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

    def update_plot(self, partial_accuracy, num_instances):
        self.visualizer.on_new_train_step(partial_accuracy, num_instances)
        pass

    def update_metrics(self):
        self.global_accuracy = ((self.global_sample_count - self.partial_sample_count) / self.global_sample_count) * \
                               self.global_accuracy + (self.partial_sample_count / self.global_sample_count) * \
                                                      (self.partial_correct_predicts/self.partial_sample_count)
        if self.show_plot:
            self.update_plot(self.partial_correct_predicts/self.partial_sample_count, self.global_sample_count)
        self._reset_partials()

    def _reset_partials(self):
        self.partial_sample_count = 0
        self.partial_correct_predicts = 0

    def _reset_globals(self):
        self.global_sample_count = 0
        self.global_correct_predicts = 0
        self.global_accuracy = 0.0

    def start_plot(self, n_wait, dataset_name):
        self.visualizer = EvaluationVisualizer(n_wait=n_wait, dataset_name=dataset_name)
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
            elif name == 'show_plot':
                self.show_plot = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value