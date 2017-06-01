__author__ = "Guilherme Matsumoto"

from skmultiflow.data.CsvFileStream import CsvFileStream
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.classification.NaiveBayes import NaiveBayes
from skmultiflow.options.FileOption import FileOption
import logging

def demo():
    #opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    #stream = CsvFileStream(opt, 7)
    #stream = WaveformGenerator()
    stream = RandomTreeGenerator()
    stream.prepareForUse()
    classifier = NaiveBayes()
    classes = [0, 1]
    pretrain = True
    instCount = 0
    correctPredict = 0
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    for i in range(50001):
        if pretrain:
            logging.info('Learning model on 1000 instances')

            X, y = stream.nextInstance(1000)
            classifier.partial_fit(X, y, classes, pretrain)
            logging.info('Evaluating...')
        else:
            if (i % 2500) == 0:
                msg = str(((i+1)//2500)*5) + '%'
                logging.info(msg)
            X, y = stream.nextInstance()
            instCount += 1
            predict = classifier.predict(X)
            #print("prediction: " + str(predict[0]) + " | real: " + str(y[0]))
            if predict[0] == y[0]:
                correctPredict += 1
            classifier.partial_fit(X, y, classes, pretrain)

        pretrain = False
    msg = 'Global accuracy: ' + str(correctPredict/instCount)
    logging.info(msg)
    msg = 'Total instances: ' + str(instCount)
    logging.info(msg)
    pass

if __name__ == "__main__":
    demo()