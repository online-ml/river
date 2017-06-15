__author__ = 'Guilherme Matsumoto'

from skmultiflow.visualization.BaseListener import BaseListener
from skmultiflow.core.utils.data_structures import FastBuffer
import numpy as np
import matplotlib.pyplot as plt
import warnings


class EvaluationVisualizer(BaseListener):
    def __init__(self, n_wait = 200, dataset_name = 'Unnamed graph', show_kappa=False):
        super().__init__()
        #default values
        self.temp = []
        self.true_labels = None
        self.predictions = None
        self.partial_performance = None
        self.global_performance = None
        self.partial_kappa = None
        self.global_kappa = None
        self.n_wait = None
        self.dataset_name = None
        self.show_kappa = None
        self.line_global_performance = None
        self.line_partial_performance = None
        self.line_global_kappa = None
        self.line_partial_kappa = None
        self.num_plots = 0

        self.configure(n_wait, dataset_name, show_kappa)

    def on_new_train_step(self, performance_point, train_step):
        if (train_step % self.n_wait == 0):
            self.draw(performance_point, train_step)
        pass

    def on_new_data(self, true_labels, predictions):
        #flattened_labels = np.ravel(true_labels)
        #flattened_preds = np.ravel(predictions)
        '''
        for i in range(len(true_labels)):
            self.true_labels.add_element(true_labels[i])
            self.predictions.add_element(predictions[i])
        '''
        pass


    def configure(self, n_wait, dataset_name, show_kappa):
        self.X = []
        self.partial_performance = []
        self.global_performance = []
        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.show_kappa = show_kappa
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        #plt.title(dataset_name)
        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        #plt.figure.__name__ = dataset_name
        self.fig.suptitle(dataset_name)
        #fig = plt.gcf()
        self.num_plots = 1 + self.show_kappa
        base = 11 + self.num_plots*100
        self.fig.canvas.set_window_title('scikit-multiflow')

        self.subplot_performance = self.fig.add_subplot(base)
        if self.show_kappa:
            self.subplot_kappa = self.fig.add_subplot(base+1)
            self.subplot_kappa.set_ylabel('Kappa statistic')
            self.subplot_kappa.set_xlabel('Samples analyzed')

        self.subplot_performance.set_ylabel('Performance ratio')
        self.subplot_performance.set_xlabel('Samples analyzed')

        self.line_partial_performance, = self.subplot_performance.plot(self.X, self.partial_performance, label='Partial performance (last 200 samples)')
        self.line_global_performance, = self.subplot_performance.plot(self.X, self.global_performance, label='Global performance')
        self.subplot_performance.legend(handles=[self.line_global_performance, self.line_partial_performance])
        self.subplot_performance.set_ylim([0,1])

        if self.show_kappa:
            self.partial_kappa = []
            self.global_kappa = []
            self.true_labels = FastBuffer(n_wait)
            self.predictions = FastBuffer(n_wait)
            self.line_partial_kappa, = self.subplot_kappa.plot(self.X, self.partial_kappa, label='Partial Kappa (last 200 samples)')
            self.line_global_kappa, = self.subplot_kappa.plot(self.X, self.global_kappa, label='Global Kappa')
            self.subplot_kappa.legend(handles=[self.line_global_kappa, self.line_partial_kappa])
            self.subplot_kappa.set_ylim([-1,1])
            self.true_labels = FastBuffer(self.n_wait)
            self.predictions = FastBuffer(self.n_wait)

    def draw(self, new_performance_point, train_step):
        self.X.append(train_step)
        for i in range(len(new_performance_point)):
            if i == 0:
                self.partial_performance.append(new_performance_point[i])
                self.global_performance.append(np.mean(self.partial_performance))
                self.line_partial_performance.set_data(self.X, self.partial_performance)
                self.line_global_performance.set_data(self.X, self.global_performance)
            if i == 1:
                self.partial_kappa.append(new_performance_point[i])
                self.global_kappa.append(np.mean(self.partial_kappa))
                self.line_partial_kappa.set_data(self.X, self.partial_kappa)
                self.line_global_kappa.set_data(self.X, self.global_kappa)
        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []
        self.temp.append(self.subplot_performance.annotate('Partial: ' + str(round(new_performance_point[0], 3)), xy=(train_step, new_performance_point[0]),
                                      xytext=(8, 0), textcoords = 'offset points'))
        self.temp.append(self.subplot_performance.annotate('Global: ' + str(round(self.global_performance[len(self.global_performance)-1], 3)),
                                      xy=(train_step, self.global_performance[len(self.global_performance) - 1]),
                                      xytext=(8, 0), textcoords = 'offset points'))
        if self.show_kappa:
            self.temp.append(self.subplot_kappa.annotate('Partial: ' + str(round(new_performance_point[1], 3)), xy=(train_step, new_performance_point[1]),
                                                         xytext=(8,0), textcoords='offset points'))
            self.temp.append(self.subplot_kappa.annotate('Global: ' + str(round(self.global_kappa[len(self.global_kappa)-1], 3)),
                                                         xy=(train_step, self.global_kappa[len(self.global_kappa)-1]),
                                                         xytext=(8,0), textcoords='offset points'))

        self.subplot_performance.set_xlim([np.min(self.X), 1.2*np.max(self.X)])
        self.subplot_performance.set_ylim([0,1])
        if self.show_kappa:
            self.subplot_kappa.set_xlim([np.min(self.X), 1.2*np.max(self.X)])
            self.subplot_kappa.set_ylim([-1,1])
        #plt.xlim([np.min(self.X), 520000])
        plt.draw()
        plt.pause(0.00001)

    def hold(self):
        plt.show(block=True)
        pass

if __name__ == "__main__":
    ev = EvaluationVisualizer()
    print(ev.get_class_type())