__author__ = 'Guilherme Matsumoto'

import time
import warnings
from skmultiflow.visualization.base_listener import BaseListener
from skmultiflow.core.utils.data_structures import FastBuffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class EvaluationVisualizer(BaseListener):
    def __init__(self, n_wait = 200, dataset_name = 'Unnamed graph', plots=None):
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
        self.scatter_true_labels = None
        self.scatter_predicts = None
        self.scatter_true_labels_colors = None
        self.scatter_predicts_colors = None

        #configs
        self.n_wait = None
        self.dataset_name = None

        #lines
        self.line_global_performance = None
        self.line_partial_performance = None

        self.line_global_kappa = None
        self.line_partial_kappa = None

        self.line_scatter_predicts = None
        self.line_scatter_true_labels = None

        #show configs
        self.num_plots = 0

        #subplot default
        self.subplot_kappa = None
        self.subplot_performance = None
        self.subplot_scatter_points = None

        if plots is not None:
            if len(plots) < 1:
                raise ValueError('No plots were give.')
            else:
                self.configure(n_wait, dataset_name, plots)
        else:
            raise ValueError('No plots were given.')

    def on_new_train_step(self, train_step, dict):
        if (train_step % self.n_wait == 0):
            try:
                self.draw(train_step, dict)
                #self.draw_scatter_points(train_step, dict)
            except BaseException as exc:
                raise ValueError('Wrong data format.')
        pass

    def on_new_scatter_data(self, X, y, prediction):
        #self.draw_scatter_points(X, y, prediction)
        pass


    def configure(self, n_wait, dataset_name, plots):
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        warnings.filterwarnings("ignore", ".*left==right.*")

        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.plots = plots
        self.X = []

        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle(dataset_name)
        self.num_plots = len(self.plots)
        base = 11 + self.num_plots * 100
        self.fig.canvas.set_window_title('scikit-multiflow')

        if 'performance' in self.plots:
            self.partial_performance = []
            self.global_performance = []

            self.subplot_performance = self.fig.add_subplot(base)
            self.subplot_performance.set_title('Classifier\'s accuracy')
            self.subplot_performance.set_ylabel('Performance ratio')
            self.subplot_performance.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_performance, = self.subplot_performance.plot(self.X, self.partial_performance,
                                                                           label='Partial performance (last ' + str(
                                                                               self.n_wait) + ' samples)')
            self.line_global_performance, = self.subplot_performance.plot(self.X, self.global_performance,
                                                                          label='Global performance')
            self.subplot_performance.legend(handles=[self.line_partial_performance, self.line_global_performance])
            self.subplot_performance.set_ylim([0, 1])


        if 'kappa' in self.plots:
            self.partial_kappa = []
            self.global_kappa = []

            self.subplot_kappa = self.fig.add_subplot(base)
            self.subplot_kappa.set_title('Classifier\'s Kappa')
            self.subplot_kappa.set_ylabel('Kappa statistic')
            self.subplot_kappa.set_xlabel('Samples analyzed')
            base += 1

            self.global_kappa = []
            self.partial_kappa = []
            self.line_global_kappa, = self.subplot_kappa.plot(self.X, self.global_kappa, label='Global kappa')
            self.line_partial_kappa, = self.subplot_kappa.plot(self.X, self.partial_kappa,
                                                               label='Sliding window Kappa (last '
                                                                     + str(self.n_wait) + ' samples)')
            self.subplot_kappa.legend(handles=[self.line_partial_kappa, self.line_global_kappa])
            self.subplot_kappa.set_ylim([-1, 1])


        if 'scatter' in self.plots:
            self.scatter_predicts = []
            self.scatter_true_labels = []
            self.scatter_x = []
            self.subplot_scatter_points = self.fig.add_subplot(base)
            base += 1

            self.subplot_scatter_points.set_title('Predicts and true labels')
            self.subplot_scatter_points.set_ylabel('Class labels')
            self.subplot_scatter_points.set_xlabel('Sample analyzed')
            self.scatter_true_labels_colors = []
            self.scatter_predicts_colors = []

        self.fig.subplots_adjust(hspace=.5)

        self.fig.tight_layout(pad=2.6, w_pad=0.5, h_pad=1.0)

    def draw(self, train_step, dict):
        self.X.append(train_step)

        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []

        if 'performance' in self.plots:
            self.global_performance.append(dict['performance'][0])
            self.partial_performance.append(dict['performance'][1])
            self.line_global_performance.set_data(self.X, self.global_performance)
            self.line_partial_performance.set_data(self.X, self.partial_performance)

            self.temp.append(self.subplot_performance.annotate('Global: ' + str(round(dict['performance'][0], 3)),
                                                               xy=(train_step, dict['performance'][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
            self.temp.append(self.subplot_performance.annotate('Partial: ' + str(round(dict['performance'][1], 3)),
                                                               xy=(train_step, dict['performance'][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_performance.set_ylim([0, 1])
            self.subplot_performance.set_xlim([0, 1.2 * np.max(self.X)])


        if 'kappa' in self.plots:
            self.global_kappa.append(dict['kappa'][0])
            self.partial_kappa.append(dict['kappa'][1])
            self.line_global_kappa.set_data(self.X, self.global_kappa)
            self.line_partial_kappa.set_data(self.X, self.partial_kappa)

            self.temp.append(self.subplot_kappa.annotate('Global kappa: ' + str(round(dict['kappa'][0], 3)),
                                                         xy=(train_step, dict['kappa'][0]), xytext=(8, 0),
                                                         textcoords='offset points'))
            self.temp.append(self.subplot_kappa.annotate('Sliding window Kappa: ' + str(round(dict['kappa'][1], 3)),
                                                         xy=(train_step, dict['kappa'][1]), xytext=(8, 0),
                                                         textcoords='offset points'))
            self.subplot_kappa.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa.set_ylim([-1, 1])


        if 'scatter' in self.plots:
            self.scatter_x.append(train_step)
            self.scatter_true_labels.append(dict['scatter'][0])
            self.scatter_predicts.append(dict['scatter'][1])
            if dict['scatter'][0] == dict['scatter'][1]:
                self.scatter_predicts_colors.append('g')
                self.scatter_true_labels_colors.append('g')
            else:
                self.scatter_predicts_colors.append('r')
                self.scatter_true_labels_colors.append('y')

            classes = np.unique([self.scatter_predicts, self.scatter_true_labels])
            scat_true = self.subplot_scatter_points.scatter(self.scatter_x, self.scatter_true_labels, s=6,
                                                                label='True labels', c=self.scatter_true_labels_colors)
            scat_pred = self.subplot_scatter_points.scatter(self.scatter_x, self.scatter_predicts, s=6,
                                                                label='Predicts', c=self.scatter_predicts_colors)
            self.subplot_scatter_points.set_xlim(np.min(self.scatter_x), 1.2 * np.max(self.scatter_x))
            self.subplot_scatter_points.set_ylim(np.min(classes) - 1, np.max(classes) + 1)
            colour = ['r', 'y', 'g']
            legend = ['Prediction', 'True label', 'Correct prediction']
            recs = []
            for i in range(0, len(colour)):
                recs.append(mpatches.Circle((0, 0), 1, fc=colour[i]))
            #self.subplot_scatter_points.legend(handles=[scat_true, scat_pred])
            self.subplot_scatter_points.legend(recs, legend, loc=4)

        plt.draw()
        plt.pause(0.00001)


    def draw_scatter_points(self, X, y, predict):
        pass

    def hold(self):
        plt.show(block=True)

    def get_info(self):
        pass

if __name__ == "__main__":
    ev = EvaluationVisualizerNew()
    print(ev.get_class_type())