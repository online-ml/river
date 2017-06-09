__author__ = 'Guilherme Matsumoto'

import skmultiflow.tasks.testMain as t
import skmultiflow.demos.streamPlusClassifier as spc
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
from skmultiflow.core.pipeline.Pipeline import Pipeline
from skmultiflow.classification.Perceptron import PerceptronMask


def demo(argv):
    t.demo_file_stream()
    #t.demo_waveform_gen(argv)
    #t.demo_random_tree_gen(argv)
    return None

def testPreq(argv):
    t.demo_preq(argv)

def data_gen():
    while True:
        yield np.random.rand(10)

def demoSCP():
    spc.demo()
    pass

def demo_pipeline():
    pass


if __name__ == '__main__':
    #plt.interactive(False)
    #print(sys.argv)
    #demo(sys.argv[1:])
    #testPreq(sys.argv)
    #testPlot()
    #testIncrementalPlot()
    demoSCP()
