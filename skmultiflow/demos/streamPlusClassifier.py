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
from timeit import default_timer as timer


def demo():
    # First demo, NOT TO BE USED
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    optList = [['-c', '2'], ['-o', '0'], ['-u', '5']]

    stream = FileStream(opt, 7)
    #stream = WaveformGenerator()
    #stream = RandomTreeGenerator(optList)
    #stream = SEAGenerator(classificationFunction=1, instance_seed=67, balanceClasses=False, noisePercentage=0.82)
    #stream = RandomRBFGenerator()
    #stream = RandomRBFGeneratorDrift()
    stream.prepare_for_use()
    print(stream.get_classes())

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    msg = 'Generating ' + str(stream.get_num_classes()) + ' classes'
    logging.info(msg)

    #visualizer = EvaluationVisualizer(n_wait=200, dataset_name='Cover Type - 7 class labels')
    visualizer = EvaluationVisualizer(n_wait=200, dataset_name=stream.get_plot_name())

    #classifier = NaiveBayes()
    classifier = PerceptronMask()

    #the commented classes is for the NaiveBayes classifier
    classes = stream.get_classes()

    pretrain = True
    inst_count = 0
    correct_predict = 0
    partial_count = 0
    partial_correct_predict = 0

    x_length = 100000
    n_wait = 200

    if pretrain:
        logging.info('Learning model on 1000 instances')
        X, y = stream.next_instance(1000)
        classifier.partial_fit(X, y, classes)
        logging.info('Evaluating...')
    init_time = timer()
    for i in range(1, x_length+1):
        if (i % n_wait) == 0:
            #msg = 'Partial performance (over 200 samples): ' + str(partial_correct_predict / partial_count)
            #logging.info(msg)
            # plt.scatter(i, partial_correct_predict / partial_count)
            # plt.draw()
            visualizer.on_new_train_step(partial_correct_predict / partial_count, i)
            partial_correct_predict = 0
            partial_count = 0
        if (i % (x_length/20)) == 0:
            msg = str(((i) // (x_length/20))*5) + '%'
            logging.info(msg)
        X, y = stream.next_instance()
        # X = X.reshape(1, -1)
        inst_count += 1
        partial_count += 1
        predict = classifier.predict(X)
        # print("prediction: " + str(predict[0]) + " | real: " + str(y[0]))
        if predict[0] == y[0]:
            correct_predict += 1
            partial_correct_predict += 1
        classifier.partial_fit(X, y, classes)

    msg = 'Evaluation time: ' + str(round(timer() - init_time), 3)
    logging.info(msg)
    msg = 'Global accuracy: ' + str(round(correct_predict/inst_count), 3)
    logging.info(msg)
    msg = 'Total instances: ' + str(inst_count)
    logging.info(msg)

    #keep the graph open
    visualizer.hold()
    pass

if __name__ == "__main__":
    demo()