__author__ = 'Guilherme Matsumoto'

import skmultiflow.tasks.testMain as t
import skmultiflow.demos.streamPlusClassifier as spc
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random


def demo(argv):
    t.demo_file_stream()
    #t.demo_waveform_gen(argv)
    #t.demo_random_tree_gen(argv)
    return None

def testPreq(argv):
    t.demo_preq(argv)

def testIncrementalPlot():
    x = np.linspace(0, 100, 100)
    y = np.random.randint(60, 100, 100)
    count = 100
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, color='k')

    def update(num, x, y, line):
        line.set_data(x[:num], y[:num])
        line.axes.axis([0, 100, 40, 110])
        count += 1
        return line,

    ani = animation.FuncAnimation(fig, update, count, fargs=[x, y, line],
                                  interval=25, blit=False)
    plt.show()

def testPlot():
    ysample = np.random.randint(60, 100, 100)

    xdata = []
    ydata = []

    plt.show()

    axes = plt.gca()
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    line, = axes.plot(xdata, ydata, 'r-')

    for i in range(100):
        xdata.append(i)
        ydata.append(ysample[i])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.5)

    # add this if you don't want the window to disappear at the end
    plt.show()

def update(data, line, count):
    line.set_ydata(data)
    return line

def data_gen():
    while True:
        yield np.random.rand(10)

def demoSCP():
    spc.demo()
    pass

if __name__ == '__main__':
    #plt.interactive(False)
    #print(sys.argv)
    #demo(sys.argv[1:])
    #testPreq(sys.argv)
    #testPlot()
    #testIncrementalPlot()
    demoSCP()
