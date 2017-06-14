__author__ = 'Guilherme Matsumoto'

from skmultiflow.visualization.BaseListener import BaseListener
from skmultiflow.core.utils.data_structures import FastBuffer
from sklearn.metrics import cohen_kappa_score
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
        self.line_kappa_statistic = None
        self.queue = None

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
        '''
        if self.show_kappa:
            self.partial_kappa = []
            self.global_kappa = []
            self.queue = FastBuffer(n_wait)
            self.line_partial_kappa, = plt.plot(self.X, self.partial_kappa, label='Partial Kappa (last 200 samples)')
            self.line_global_kappa, = plt.plot(self.X, self.global_kappa, label='Global Kappa')
            plt.legend(handles=[self.line_global_performance, self.line_partial_performance,
                                self.line_global_kappa, self.line_partial_kappa])
            plt.ylim([-1,1])
            self.true_labels = FastBuffer(self.n_wait)
            self.predictions = FastBuffer(self.n_wait)
        else:
            plt.legend(handles=[self.line_global_performance, self.line_partial_performance])
            plt.ylim([0,1])
        '''
        plt.legend(handles=[self.line_global_performance, self.line_partial_performance])
        plt.ylim([0, 1])
        pass

    def draw(self, new_performance_point, train_step):
        self.X.append(train_step)
        self.partial_performance.append(new_performance_point)
        self.global_performance.append(np.mean(self.partial_performance))
        self.line_partial_performance.set_data(self.X, self.partial_performance)
        self.line_global_performance.set_data(self.X, self.global_performance)
        '''
        if self.show_kappa:
            kappa_stat = cohen_kappa_score(self.true_labels.get_queue(), self.predictions.get_queue())
            self.partial_kappa.append(kappa_stat)
            global_kappa_stat = np.mean(self.partial_kappa)
            self.global_kappa.append(global_kappa_stat)
            self.line_partial_performance.set_data(self.X, self.partial_kappa)
            self.line_global_performance.set_data(self.X, self.global_kappa)
        '''
        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []
        self.temp.append(plt.annotate('Partial: ' + str(round(new_performance_point, 3)), xy=(train_step, new_performance_point),
                                      xytext=(8, 0), textcoords = 'offset points'))
        self.temp.append(plt.annotate('Global: ' + str(round(self.global_performance[len(self.global_performance)-1], 3)),
                                      xy=(train_step, self.global_performance[len(self.global_performance) - 1]),
                                      xytext=(8, 0), textcoords = 'offset points'))
        '''
        if self.show_kappa:
            self.temp.append(
                plt.annotate('Partial kappa: ' + str(round(kappa_stat, 3)), xy=(train_step, kappa_stat),
                             xytext=(8, 0), textcoords='offset points'))
            self.temp.append(
                plt.annotate('Global kappa: ' + str(round(global_kappa_stat, 3)),
                             xy=(train_step, global_kappa_stat),
                             xytext=(8, 0), textcoords='offset points'))
        '''
        plt.xlim([np.min(self.X), 1.2*np.max(self.X)])
        #plt.xlim([np.min(self.X), 520000])
        plt.draw()
        plt.pause(0.00001)

    def hold(self):
        plt.show(block=True)
        pass

if __name__ == "__main__":
    ev = EvaluationVisualizer()
    print(ev.get_class_type())