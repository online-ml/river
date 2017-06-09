__author__ = "Guilherme Matsumoto"

from skmultiflow.data.FileStream import FileStream
from skmultiflow.data.FileToStream import FileToStream
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.data.generators.SEAGenerator import SEAGenerator
from skmultiflow.data.generators.RandomRBFGenerator import RandomRBFGenerator
from skmultiflow.data.generators.RandomRBFGeneratorDrift import RandomRBFGeneratorDrift
from skmultiflow.classification.NaiveBayes import NaiveBayes
from skmultiflow.classification.Perceptron import PerceptronMask
from skmultiflow.options.FileOption import FileOption
from skmultiflow.visualization.EvaluationVisualizer import EvaluationVisualizer
import logging


def demo():
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    optList = [['-c', '2'], ['-o', '0'], ['-u', '5']]

    stream = FileStream(opt, 7)
    #stream = WaveformGenerator()
    #stream = RandomTreeGenerator(optList)
    #stream = SEAGenerator(classificationFunction=1, instanceSeed=67, balanceClasses=False, noisePercentage=0.82)
    #stream = RandomRBFGenerator()
    #stream = RandomRBFGeneratorDrift()
    stream.prepareForUse()
    print(stream.getClasses())

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    msg = 'Generating ' + str(stream.getNumClasses()) + ' classes'
    logging.info(msg)

    #visualizer = EvaluationVisualizer(n_wait=200, dataset_name='Cover Type - 7 class labels')
    visualizer = EvaluationVisualizer(n_wait=200, dataset_name=stream.getPlotName())

    #classifier = NaiveBayes()
    classifier = PerceptronMask()

    #the commented classes is for the NaiveBayes classifier
    classes = stream.getClasses()

    pretrain = True
    instCount = 0
    correctPredict = 0
    partialCount = 0
    partialCorrectPredict = 0

    x_length = 50000
    n_wait = 200

    if pretrain:
        logging.info('Learning model on 1000 instances')
        X, y = stream.nextInstance(1000)
        classifier.partial_fit(X, y, classes, pretrain)
        logging.info('Evaluating...')

    for i in range(1, x_length+1):
        if (i % n_wait) == 0:
            #msg = 'Partial performance (over 200 samples): ' + str(partialCorrectPredict / partialCount)
            #logging.info(msg)
            # plt.scatter(i, partialCorrectPredict / partialCount)
            # plt.draw()
            visualizer.onNewTrainStep(partialCorrectPredict / partialCount, i)
            partialCorrectPredict = 0
            partialCount = 0
        if (i % (x_length/20)) == 0:
            msg = str(((i) // (x_length/20))*5) + '%'
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