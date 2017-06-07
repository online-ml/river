__author__ = "Guilherme Matsumoto"

from skmultiflow.data.CsvFileStream import CsvFileStream
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.classification.NaiveBayes import NaiveBayes
from skmultiflow.classification.Perceptron import PerceptronMask
from skmultiflow.options.FileOption import FileOption
from skmultiflow.visualization.EvaluationVisualizer import EvaluationVisualizer
from sklearn import preprocessing
import logging
import matplotlib.pyplot as plt

def demo():
    #opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    #stream = CsvFileStream(opt, 7)
    #stream = WaveformGenerator()
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    optList = [['-c', '2'], ['-o', '5'], ['-u', '5']]
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    #stream = RandomTreeGenerator(optList)
    stream = CsvFileStream(opt, 7)
    stream.prepareForUse()
    msg = 'Generating ' + str(stream.getNumClasses()) + ' classes'
    logging.info(msg)

    #plotting tests
    #plt.axis([0, stream.estimatedRemainingInstances(), 0, 1])
    #plt.ion()
    visualizer = EvaluationVisualizer(n_wait=200, dataset_name='Cover Type - 7 class labels')

    #classifier = NaiveBayes()
    classifier = PerceptronMask()

    #the commented classes is for the NaiveBayes classifier
    #classes = [0, 1]
    classes = [1, 2, 3, 4, 5, 6, 7]

    pretrain = True
    instCount = 0
    correctPredict = 0
    partialCount = 0
    partialCorrectPredict = 0

    if pretrain:
        logging.info('Learning model on 1000 instances')
        X, y = stream.nextInstance(1000)
        classifier.partial_fit(X, y, classes, pretrain)
        logging.info('Evaluating...')

    for i in range(1, 500001):
        if (i % 200) == 0:
            #msg = 'Partial performance (over 200 samples): ' + str(partialCorrectPredict / partialCount)
            #logging.info(msg)
            # plt.scatter(i, partialCorrectPredict / partialCount)
            # plt.draw()
            visualizer.onNewTrainStep(partialCorrectPredict / partialCount, i)
            partialCorrectPredict = 0
            partialCount = 0
        if (i % 5000) == 0:
            msg = str(((i) // 5000)) + '%'
            logging.info(msg)
        X, y = stream.nextInstance()
        # X = X.reshape(1, -1)
        instCount += 1
        partialCount += 1
        predict = classifier.predict(X)
        # print("prediction: " + str(predict[0]) + " | real: " + str(y[0]))
        if predict[0] == y[0]:
            correctPredict += 1
            partialCorrectPredict += 1
        classifier.partial_fit(X, y, classes, pretrain)

    msg = 'Global accuracy: ' + str(correctPredict/instCount)
    logging.info(msg)
    msg = 'Total instances: ' + str(instCount)
    logging.info(msg)

    #keep the graph open
    visualizer.hold()
    pass

if __name__ == "__main__":
    demo()