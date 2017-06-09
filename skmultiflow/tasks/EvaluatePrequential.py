__author__ = 'Guilherme Matsumoto'

from skmultiflow.tasks.BaseTask import BaseTask
from skmultiflow.data.StreamCreator import *
import sys, argparse
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# stream import
from skmultiflow.data.FileStream import FileStream
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator

# classifier import
from skmultiflow.classification.NaiveBayes import NaiveBayes


class EvaluatePrequential(BaseTask):

    def __init__(self, argv):
        super().__init__()
        # default values
        self.n_wait = 200
        self.display_metric = 200
        self.data_stream = None
        self.classifier = None
        self.max_instances = 100000
        self.max_time = 1000
        self.model_size_limit = -1
        self.output_file = None
        self.num_instances = 0
        self.global_accuracy = 0
        self.plot = None
        self.configure(argv)
        pass

    def configure(self, argv):
        args = vars(self.parse(argv))
        CreateStreamFromArgumentDict(args['stream'][:])

        pass

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


    def do_task(self, stream = None, classifier = None):
        self.create_plot()
        self.train_and_test(stream, classifier)
        pass

    def train_and_test(self, stream = None, classifier = None):
        init_time = timer()
        end_time = timer()
        partial_num_instances = 0
        partial_correct_predictions = 0
        inst = None
        while((self.num_instances < self.max_instances) & ((end_time - init_time) < self.max_time)):
            inst = stream.next_instance()
            if classifier.predict(inst[:stream.get_num_attributes()]) == inst[stream.get_num_attributes:]:
                partial_correct_predictions += 1
            classifier.partial_fit(inst[:stream.get_num_attributes()], inst[stream.get_num_attributes():])

            partial_num_instances += 1
            self.num_instances += 1
            if ((partial_num_instances >= self.n_wait) | (self.num_instances == self.max_instances - 1)):
                self.update_plot()
                self.update_results(partial_correct_predictions, partial_num_instances)
                partial_num_instances = 0
                partial_correct_predictions = 0


        end_time = timer()
        print("Test and train time: " + str(end_time - init_time))

        pass

    def update_plot(self, accuracy, num_instances):

        pass

    def update_results(self, partial_accuracy, partial_num_instances):
        self.global_accuracy = ((self.num_instances - partial_num_instances) / self.num_instances) * self.global_accuracy + (partial_num_instances / self.num_instances) * partial_accuracy
        pass

    def create_plot(self):
        plt.show()
        axes = plt.gca()
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        line, = axes.plot(xdata, ydata, 'r-')
        pass