__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.FileStream import FileStream
from skmultiflow.options.FileOption import FileOption
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from skmultiflow.tasks.EvaluatePrequential import EvaluatePrequential
from timeit import default_timer as timer
import sys, getopt, logging

def demo_file_stream():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    start = timer()
    opt = FileOption("FILE", "OPT_NAME", "skmultiflow/datasets/covtype.csv", "CSV", False)
    t = FileStream(opt, 7)
    t.prepareForUse()
    end = timer()
    print("Configuration time: " + str(end-start))

    start = timer()
    for i in range(1):
    #while (t.hasMoreInstances()):
        X, y = t.nextInstance(4)
        logging.info('this is X:')
        logging.info(X)
        logging.info('this is y:')
        logging.info(y)
        #print(str(y[0]))
        #n.toString()
    end = timer()
    print("CSV - Next Instance time: " + str(end-start))
    return None

def demo_waveform_gen(argv):
    #print(argv)
    try:
        opts, args = getopt.getopt(argv, "i:n")
    except getopt.GetoptError:
        print ("testMain.py -i <Seed for random generation of instances> -n (Adds noise)")
        sys.exit(2)

    optList = []
    for opt, arg in opts:
        optList.append([opt, arg])


    wfg = WaveformGenerator()
    wfg.prepareForUse()

    i = 0
    start = timer()
    #while(wfg.hasMoreInstances()):
    for i in range(10):
        X, y = wfg.nextInstance(2)
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
        print("usage: testMain.py -r <Seed for random generation of tree> -i <Seed for random generation of instances>"
              " -c <The number of classes to generate> -o <The number of nominal attributes to generate>"
              " -u <The number of numerical attributes to generate> -v <The number of values to generate per nominal attribute>"
              " -d <The maximum depth of the tree concept> -l <The first level of the tree above MaxTreeDepth that can have leaves>"
              " -f <The fraction of leaves per level from FirstLeafLevel onwards>")
        sys.exit(2)

    optList = []
    for opt, arg in opts:
        optList.append([opt, arg])

    rtg = RandomTreeGenerator(optList)
    rtg.prepareForUse()

    i = 0
    start = timer()
    # while(wfg.hasMoreInstances()):
    for i in range(20):
        X, y = rtg.nextInstance()
        #print(X)
        print(y)
        #o.toString()
    end = timer()
    print("Random Tree - Generation time: " + str(end - start))

    return None

def demo_preq(argv):
    t = EvaluatePrequential(argv)
    pass

if __name__ == '__main__':
    demo_file_stream()
    #demo_waveform_gen(sys.argv[1:])
    #demo_random_tree_gen(sys.argv[1:])

