__author__ = 'Guilherme Matsumoto'

from skmultiflow.visualization.BaseListener import BaseListener
import numpy as np
import matplotlib.pyplot as plt
import warnings


class EvaluationVisualizer(BaseListener):
    def __init__(self, n_wait = 200, dataset_name = 'Unnamed graph'):
        super().__init__()
        self.temp = []
        self.configure(n_wait, dataset_name)

    def onNewTrainStep(self, performancePoint, trainStep):
        if (trainStep % self.n_wait == 0):
            self.draw(performancePoint, trainStep)
        pass

    def configure(self, n_wait, dataset_name):
        self.X = []
        self.partial_performance = []
        self.global_performance = []
        self.n_wait = n_wait
        self.datasetName = dataset_name
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        plt.title(dataset_name)
        plt.ion()
        plt.figure.__name__ = dataset_name
        fig = plt.gcf()
        fig.canvas.set_window_title('scikit-multiflow')
        plt.ylabel('Performance ratio')
        plt.xlabel('Samples analyzed')
        #self.fig = plt.figure(figsize=(16, 8))
        self.linePartialPerformance, = plt.plot(self.X, self.partial_performance, label='Partial performance (last 200 samples)')
        self.lineGlobalPerformance, = plt.plot(self.X, self.global_performance, label='Global performance')
        plt.legend(handles=[self.lineGlobalPerformance, self.linePartialPerformance])
        plt.ylim([0,1])
        pass

    def draw(self, newPerformancePoint, trainStep):
        self.X.append(trainStep)
        self.partial_performance.append(newPerformancePoint)
        self.global_performance.append(np.mean(self.partial_performance))
        self.linePartialPerformance.set_data(self.X, self.partial_performance)
        self.lineGlobalPerformance.set_data(self.X, self.global_performance)
        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []
        self.temp.append(plt.annotate('Partial: ' + str(round(newPerformancePoint, 3)), xy=(trainStep, newPerformancePoint),
                                      xytext=(8, 0), textcoords = 'offset points'))
        self.temp.append(plt.annotate('Global: ' + str(round(self.global_performance[len(self.global_performance)-1], 3)),
                                      xy=(trainStep, self.global_performance[len(self.global_performance)-1]),
                                      xytext=(8, 0), textcoords = 'offset points'))
        plt.xlim([np.min(self.X), 1.2*np.max(self.X)])
        #plt.xlim([np.min(self.X), 520000])
        plt.draw()
        plt.pause(0.0001)

    def hold(self):
        plt.show(block=True)
        pass
