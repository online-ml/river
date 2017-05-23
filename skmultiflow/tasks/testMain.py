__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.CsvFileStream import CsvFileStream
from skmultiflow.options.FileOption import FileOption
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator
from timeit import default_timer as timer
import sys, getopt

def demo_file_stream():
    start = timer()
    opt = FileOption("FILE", "OPT_NAME", "../datasets/covtype.csv", "CSV", False)
    t = CsvFileStream(opt, 54)
    t.prepareForUse()
    end = timer()
    print("Configuration time: " + str(end-start))

    start = timer()
    for i in range(100000):
    #while (t.hasMoreInstances()):
        n = t.nextInstance()
        #n.toString()
    end = timer()
    print("Next Instance time: " + str(end-start))
    return None

def demo_waveform_gen(argv):
    print(argv)
    try:
        opts, args = getopt.getopt(argv, "i:n")
    except getopt.GetoptError:
        print ("testMain.py -i <Seed for random generation of instances> -n (Adds noise)")
        sys.exit(2)

    optList = []
    for opt, arg in opts:
        optList.append([opt, arg])


    wfg = WaveformGenerator(optList)
    wfg.prepareForUse()

    i = 0
    start = timer()
    #while(wfg.hasMoreInstances()):
    for i in range(20):
        o = wfg.nextInstance()
        o.toString()
    end = timer()
    print("Generation time: " + str(end-start))
    return None

def demo_random_tree_gen(argv):
    print(argv)
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
    for i in range(2):
        o = rtg.nextInstance()
        o.toString()
    end = timer()
    print("Generation time: " + str(end - start))

    return None

if __name__ == '__main__':
    #demo_file_stream()
    demo_waveform_gen(sys.argv[1:])
    #demo_random_tree_gen(sys.argv[1:])

