__author__ = 'Guilherme Matsumoto'

import time
from skmultiflow.visualization.base_listener import BaseListener
from skmultiflow.core.utils.data_structures import FastBuffer
import numpy as np
import matplotlib.pyplot as plt
import warnings


class EvaluationVisualizer(BaseListener):
    def __init__(self, n_wait = 200, dataset_name = 'Unnamed graph', show_performance=True, show_kappa=False,
                 track_global_kappa=False, show_scatter_points=False):
        super().__init__()
        #default values
        self.X = None
        self.scatter_x = None
        self.temp = []
        self.true_labels = None
        self.predictions = None
        self.partial_performance = None
        self.global_performance = None
        self.global_kappa = None
        self.partial_kappa = None
        self.n_wait = None
        self.dataset_name = None
        self.scatter_true_labels = None
        self.scatter_predicts = None
        self.scatter_true_labels_colors = None
        self.scatter_predicts_colors = None
        #lines
        self.line_global_performance = None
        self.line_partial_performance = None
        self.line_global_kappa = None
        self.line_partial_kappa = None
        self.line_scatter_predicts = None
        self.line_scatter_true_labels = None
        #show configs
        self.num_plots = 0
        self.show_kappa = None
        self.track_global_kappa = None
        self.show_scatter_points = None
        self.show_performance = None
        #subplot default
        self.subplot_kappa = None
        self.subplot_performance = None
        self.subplot_scatter_points = None

        self.configure(n_wait, dataset_name, show_performance, show_kappa, track_global_kappa, show_scatter_points)

    def on_new_train_step(self, performance_point=None, train_step=None, y=None, prediction=None, global_kappa=None):
        if (train_step % self.n_wait == 0):
            if ((performance_point is not None) and (train_step is not None)):
                self.draw(performance_point, train_step, global_kappa)
            if ((y is not None) and (prediction is not None) and self.show_scatter_points):
                self.draw_scatter_points(train_step, y, prediction)
        pass

    def on_new_scatter_data(self, X, y, prediction):
        self.draw_scatter_points(X, y, prediction)
        pass


    def configure(self, n_wait, dataset_name, show_performance, show_kappa, track_global_kappa, show_scatter_points):
        self.X = []
        if show_performance:
            self.partial_performance = []
            self.global_performance = []
        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.show_performance = show_performance
        self.show_kappa = show_kappa
        self.track_global_kappa = track_global_kappa
        self.show_scatter_points = show_scatter_points
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        warnings.filterwarnings("ignore", ".*left==right.*")
        #plt.title(dataset_name)
        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        #plt.figure.__name__ = dataset_name
        self.fig.suptitle(dataset_name)
        #fig = plt.gcf()
        self.num_plots = self.show_performance + self.show_kappa + self.show_scatter_points
        base = 11 + self.num_plots*100
        self.fig.canvas.set_window_title('scikit-multiflow')

        if self.show_performance:
            self.subplot_performance = self.fig.add_subplot(base)
            self.subplot_performance.set_title('Classifier\'s accuracy')
            self.subplot_performance.set_ylabel('Performance ratio')
            self.subplot_performance.set_xlabel('Samples analyzed')
            base += 1
        if self.show_kappa:
            self.subplot_kappa = self.fig.add_subplot(base)
            self.subplot_kappa.set_title('Classifier\'s Kappa')
            self.subplot_kappa.set_ylabel('Kappa statistic')
            self.subplot_kappa.set_xlabel('Samples analyzed')
            base += 1
        if self.show_scatter_points:
            self.subplot_scatter_points = self.fig.add_subplot(base)
            self.subplot_scatter_points.set_title('Predicts and true labels')
            self.subplot_scatter_points.set_ylabel('Class labels')
            self.subplot_scatter_points.set_xlabel('Sample analyzed')
            self.scatter_true_labels_colors = []
            self.scatter_predicts_colors = []

        self.fig.subplots_adjust(hspace=.5)

        if self.show_performance:
            self.line_partial_performance, = self.subplot_performance.plot(self.X, self.partial_performance, label='Partial performance (last ' + str(self.n_wait) + ' samples)')
            self.line_global_performance, = self.subplot_performance.plot(self.X, self.global_performance, label='Global performance')
            self.subplot_performance.legend(handles=[self.line_partial_performance, self.line_global_performance])
            self.subplot_performance.set_ylim([0,1])

        if self.show_kappa:
            self.partial_kappa = []
            self.line_partial_kappa, = self.subplot_kappa.plot(self.X, self.partial_kappa, label='Sliding window Kappa (last ' + str(self.n_wait) + ' samples)')
            self.subplot_kappa.legend(handles=[self.line_partial_kappa])
            self.subplot_kappa.set_ylim([-1,1])
            self.true_labels = FastBuffer(self.n_wait)
            self.predictions = FastBuffer(self.n_wait)
            if self.track_global_kappa:
                self.global_kappa = []
                self.line_global_kappa, = self.subplot_kappa.plot(self.X, self.global_kappa, label='Global kappa')
                self.subplot_kappa.legend(handles=[self.line_partial_kappa, self.line_global_kappa])

        if self.show_scatter_points:
            self.scatter_predicts = []
            self.scatter_true_labels = []
            self.scatter_x = []

        self.fig.tight_layout(pad=2.6, w_pad=0.5, h_pad=1.0)

    def draw(self, new_performance_point, train_step, global_kappa=None):
        self.X.append(train_step)
        '''
        for i in range(len(new_performance_point)):
            if ((i == 0) & (self.show_performance)):
                self.partial_performance.append(new_performance_point[i])
                self.global_performance.append(np.mean(self.partial_performance))
                self.line_partial_performance.set_data(self.X, self.partial_performance)
                self.line_global_performance.set_data(self.X, self.global_performance)
            else:
                self.partial_kappa.append(new_performance_point[i])
                self.global_kappa.append(np.mean(self.partial_kappa))
                self.line_partial_kappa.set_data(self.X, self.partial_kappa)
                self.line_global_kappa.set_data(self.X, self.global_kappa)
            if (i == 1):
                self.partial_kappa.append(new_performance_point[i])
                self.global_kappa.append(np.mean(self.partial_kappa))
                self.line_partial_kappa.set_data(self.X, self.partial_kappa)
                self.line_global_kappa.set_data(self.X, self.global_kappa)
        '''
        if self.show_performance and self.show_kappa:
            self.partial_performance.append(new_performance_point[0])
            self.global_performance.append(np.mean(self.partial_performance))
            self.line_partial_performance.set_data(self.X, self.partial_performance)
            self.line_global_performance.set_data(self.X, self.global_performance)

            self.partial_kappa.append(new_performance_point[1])
            self.line_partial_kappa.set_data(self.X, self.partial_kappa)
            if self.track_global_kappa:
                self.global_kappa.append(global_kappa)
                self.line_global_kappa.set_data(self.X, self.global_kappa)

        elif self.show_performance and not self.show_kappa:
            self.partial_performance.append(new_performance_point[0])
            self.global_performance.append(np.mean(self.partial_performance))
            self.line_partial_performance.set_data(self.X, self.partial_performance)
            self.line_global_performance.set_data(self.X, self.global_performance)

        elif not self.show_performance and self.show_kappa:
            self.partial_kappa.append(new_performance_point[0])
            self.line_partial_kappa.set_data(self.X, self.partial_kappa)
            if self.track_global_kappa:
                self.global_kappa.append(global_kappa)
                self.line_global_kappa.set_data(self.X, self.global_kappa)

        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []
        if self.show_performance and self.show_kappa:
            self.temp.append(self.subplot_performance.annotate('Partial: ' + str(round(new_performance_point[0], 3)), xy=(train_step, new_performance_point[0]),
                                          xytext=(8, 0), textcoords = 'offset points'))
            self.temp.append(self.subplot_performance.annotate('Global: ' + str(round(self.global_performance[len(self.global_performance)-1], 3)),
                                          xy=(train_step, self.global_performance[len(self.global_performance) - 1]),
                                          xytext=(8, 0), textcoords = 'offset points'))
            self.subplot_performance.set_ylim([0, 1])
            self.subplot_performance.set_xlim([0, 1.2 * np.max(self.X)])

            self.temp.append(self.subplot_kappa.annotate('Sliding window Kappa: ' + str(round(new_performance_point[1], 3)),
                                                         xy=(train_step, new_performance_point[1]),
                                                         xytext=(8, 0), textcoords='offset points'))
            if self.track_global_kappa:
                self.temp.append(self.subplot_kappa.annotate('Global kappa: ' + str(round(global_kappa, 3)),
                                            xy=(train_step, global_kappa),
                                            xytext=(8, 0), textcoords='offset points'))
            self.subplot_kappa.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa.set_ylim([-1, 1])
        elif self.show_performance and not self.show_kappa:
            self.temp.append(self.subplot_performance.annotate('Partial: ' + str(round(new_performance_point[0], 3)),
                                                               xy=(train_step, new_performance_point[0]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.temp.append(self.subplot_performance.annotate('Global: ' +
                                                               str(round(self.global_performance[len(self.global_performance) - 1], 3)),
                                                               xy=(train_step, self.global_performance[len(self.global_performance) - 1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_performance.set_ylim([0, 1])
            self.subplot_performance.set_xlim([0, 1.2 * np.max(self.X)])

        elif not self.show_performance and self.show_kappa:
            self.temp.append(self.subplot_kappa.annotate('Sliding window Kappa: ' + str(round(new_performance_point[0], 3)), xy=(train_step, new_performance_point[0]),
                                                         xytext=(8,0), textcoords='offset points'))
            if self.track_global_kappa:
                self.temp.append(self.subplot_kappa.annotate('Global kappa: ' + str(round(global_kappa, 3)),
                                            xy=(train_step, global_kappa),
                                            xytext=(8, 0), textcoords='offset points'))
            self.subplot_kappa.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa.set_ylim([-1, 1])
        plt.draw()
        plt.pause(0.00001)

    def draw_scatter_points(self, X, y, predict):
        self.scatter_x.append(X)
        self.scatter_true_labels.append(y)
        self.scatter_predicts.append(predict)
        if y == predict:
            self.scatter_predicts_colors.append('g')
            self.scatter_true_labels_colors.append('g')
        else:
            #print('errou' + str(X))
            self.scatter_predicts_colors.append('r')
            self.scatter_true_labels_colors.append('y')
        classes = np.unique([self.scatter_predicts, self.scatter_true_labels])
        if self.subplot_scatter_points is not None:
            scat_true = self.subplot_scatter_points.scatter(self.scatter_x,self.scatter_true_labels, s=4, label='True labels', c=self.scatter_true_labels_colors)
            scat_pred = self.subplot_scatter_points.scatter(self.scatter_x, self.scatter_predicts, s=4, label='Predicts', c=self.scatter_predicts_colors)
            self.subplot_scatter_points.set_xlim(np.min(self.scatter_x), 1.2*np.max(self.scatter_x))
            self.subplot_scatter_points.set_ylim(np.min(classes)-1, np.max(classes)+1)
            self.subplot_scatter_points.legend(handles=[scat_true, scat_pred])
            plt.draw()
            plt.pause(0.0000000001)

    def hold(self):
        plt.show(block=True)

    def get_info(self):
        pass

if __name__ == "__main__":
    ev = EvaluationVisualizer()
    print(ev.get_class_type())