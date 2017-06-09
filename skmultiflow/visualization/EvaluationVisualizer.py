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

    def on_new_train_step(self, performance_point, train_step):
        if (train_step % self.n_wait == 0):
            self.draw(performance_point, train_step)
        pass

    def configure(self, n_wait, dataset_name):
        self.X = []
        self.partial_performance = []
        self.global_performance = []
        self.n_wait = n_wait
        self.dataset_name = dataset_name
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        plt.title(dataset_name)
        plt.ion()
        plt.figure.__name__ = dataset_name
        fig = plt.gcf()
        fig.canvas.set_window_title('scikit-multiflow')
        plt.ylabel('Performance ratio')
        plt.xlabel('Samples analyzed')
        #self.fig = plt.figure(figsize=(16, 8))
        self.line_partial_performance, = plt.plot(self.X, self.partial_performance, label='Partial performance (last 200 samples)')
        self.line_global_performance, = plt.plot(self.X, self.global_performance, label='Global performance')
        plt.legend(handles=[self.line_global_performance, self.line_partial_performance])
        plt.ylim([0,1])
        pass

    def draw(self, new_performance_point, train_step):
        self.X.append(train_step)
        self.partial_performance.append(new_performance_point)
        self.global_performance.append(np.mean(self.partial_performance))
        self.line_partial_performance.set_data(self.X, self.partial_performance)
        self.line_global_performance.set_data(self.X, self.global_performance)
        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []
        self.temp.append(plt.annotate('Partial: ' + str(round(new_performance_point, 3)), xy=(train_step, new_performance_point),
                                      xytext=(8, 0), textcoords = 'offset points'))
        self.temp.append(plt.annotate('Global: ' + str(round(self.global_performance[len(self.global_performance)-1], 3)),
                                      xy=(train_step, self.global_performance[len(self.global_performance) - 1]),
                                      xytext=(8, 0), textcoords = 'offset points'))
        plt.xlim([np.min(self.X), 1.2*np.max(self.X)])
        #plt.xlim([np.min(self.X), 520000])
        plt.draw()
        plt.pause(0.0001)

    def hold(self):
        plt.show(block=True)
        pass
