__author__ = 'Guilherme Matsumoto'

import getopt
import logging
import sys
from timeit import default_timer as timer

from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from sklearn.linear_model.perceptron import Perceptron


def demo_file_stream():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    start = timer()
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    t = FileStream(opt, 7)
    t.prepare_for_use()
    end = timer()
    print("Configuration time: " + str(end-start))

    start = timer()
    for i in range(1):
    #while (t.has_more_instances()):
        X, y = t.next_instance(4)
        logging.info('this is X:')
        logging.info(X)
        logging.info('this is y:')
        logging.info(y)
        #print(str(y[0]))
        #n.to_string()
    end = timer()
    print("CSV - Next Instance time: " + str(end-start))
    return None

def demo_waveform_gen(argv):
    #print(argv)
    try:
        opts, args = getopt.getopt(argv, "i:n")
    except getopt.GetoptError:
        print ("_test_main.py -i <Seed for random generation of instances> -n (Adds noise)")
        sys.exit(2)

    optList = []
    for opt, arg in opts:
        optList.append([opt, arg])


    wfg = WaveformGenerator()
    wfg.prepare_for_use()

    i = 0
    start = timer()
    #while(wfg.has_more_instances()):
    for i in range(10):
        X, y = wfg.next_instance(2)
        print(str(i))
        print(X)
        print(y)
    end = timer()
    print("Waveform - Generation time: " + str(end-start))
    return None

def demo_random_tree_gen(argv):
    #print(argv)
    try:
        opts, args = getopt.getopt(argv, "r:i:c:o:u:v:d:l:f:")
    except getopt.GetoptError:
        print("usage: _test_main.py -r <Seed for random generation of tree> -i <Seed for random generation of instances>"
              " -c <The number of classes to generate> -o <The number of nominal attributes to generate>"
              " -u <The number of numerical attributes to generate> -v <The number of values to generate per nominal attribute>"
              " -d <The maximum depth of the tree concept> -l <The first level of the tree above MaxTreeDepth that can have leaves>"
              " -f <The fraction of leaves per level from FirstLeafLevel onwards>")
        sys.exit(2)

    optList = []
    for opt, arg in opts:
        optList.append([opt, arg])

    rtg = RandomTreeGenerator(optList)
    rtg.prepare_for_use()

    i = 0
    start = timer()
    # while(wfg.has_more_instances()):
    for i in range(20):
        X, y = rtg.next_instance()
        #print(X)
        print(y)
        #o.to_string()
    end = timer()
    print("Random Tree - Generation time: " + str(end - start))

    return None

def demo_preq():
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    stream = FileStream(opt, 7)
    stream.prepare_for_use()
    classifier = Perceptron()
    eval = EvaluatePrequential(show_plot=True, pretrain_size=1000)
    eval.eval(stream=stream, classifier=classifier)
    pass

if __name__ == '__main__':
    demo_file_stream()
    #demo_waveform_gen(sys.argv[1:])
    #demo_random_tree_gen(sys.argv[1:])

