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
        self.maxInstances = 100000
        self.maxTime = 1000
        self.modelSizeLimit = -1
        self.outputFile = None
        self.numInstances = 0
        self.globalAccuracy = 0
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
        parser.add_argument("-t", dest='maxTime', type=int, help='Max number of seconds')
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


    def doTask(self, stream = None, classifier = None):
        self.createPlot()
        self.train_and_test(stream, classifier)
        pass

    def train_and_test(self, stream = None, classifier = None):
        init_time = timer()
        end_time = timer()
        partialNumInstances = 0
        partialCorrectPredictions = 0
        inst = None
        while((self.numInstances < self.maxInstances) & ((end_time - init_time) < self.maxTime)):
            inst = stream.nextInstance()
            if classifier.predict(inst[:stream.getNumAttributes()]) == inst[stream.getNumAttributes:]:
                partialCorrectPredictions += 1
            classifier.partial_fit(inst[:stream.getNumAttributes()], inst[stream.getNumAttributes():])

            partialNumInstances += 1
            self.numInstances += 1
            if ((partialNumInstances >= self.n_wait) | (self.numInstances == self.maxInstances - 1)):
                self.updatePlot()
                self.updateResults(partialCorrectPredictions, partialNumInstances)
                partialNumInstances = 0
                partialCorrectPredictions = 0


        end_time = timer()
        print("Test and train time: " + str(end_time - init_time))

        pass

    def updatePlot(self, accuracy, numInstances):

        pass

    def updateResults(self, partialAccuracy, partialNumInstances):
        self.globalAccuracy = ((self.numInstances - partialNumInstances)/self.numInstances)*self.globalAccuracy + (partialNumInstances/self.numInstances)*partialAccuracy
        pass

    def createPlot(self):
        plt.show()
        axes = plt.gca()
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        line, = axes.plot(xdata, ydata, 'r-')
        pass