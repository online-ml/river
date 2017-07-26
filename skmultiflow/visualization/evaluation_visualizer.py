__author__ = 'Guilherme Matsumoto'

import time
import warnings
import copy as cp
from skmultiflow.visualization.base_listener import BaseListener
from skmultiflow.core.utils.data_structures import FastBuffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class EvaluationVisualizer(BaseListener):
    #COLOR_MAP = ['b','c','m','y','k','w']
    COLOR_MAP = ['#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC', '#ACE600', '#D9007E', '#FFCCCC',
                 '#5E6600', '#FFFF00', '#999999', '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                 '#9BC6ED', '#915200', '#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC', '#ACE600',
                 '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999', '#FF6000', '#00FF00', '#FF00FF', '#00FFFF',
                 '#FFFF0F', '#F0CC01', '#9BC6ED', '#915200']
    def __init__(self, n_wait = 200, dataset_name = 'Unnamed graph', plots=None, n_learners=1):
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

        self.global_kappa_t = None
        self.partial_kappa_t = None

        self.global_kappa_m = None
        self.partial_kappa_m = None

        self.scatter_true_labels = None
        self.scatter_predicts = None
        self.scatter_true_labels_colors = None
        self.scatter_predicts_colors = None

        self.global_hamming_score = None
        self.partial_hamming_score = None

        self.global_hamming_loss = None
        self.partial_hamming_loss = None

        self.global_exact_match = None
        self.partial_exact_match = None

        self.global_j_index = None
        self.partial_j_index = None

        self.global_mse = None
        self.partial_mse = None

        self.global_mae = None
        self.partial_mae = None

        self.regression_true = None
        self.regression_pred = None


        #configs
        self.n_wait = None
        self.dataset_name = None
        self.n_learners = None

        #lines
        self.line_global_performance = None
        self.line_partial_performance = None

        self.line_global_kappa = None
        self.line_partial_kappa = None

        self.line_global_kappa_t = None
        self.line_partial_kappa_t = None

        self.line_global_kappa_m = None
        self.line_partial_kappa_m = None

        self.line_scatter_predicts = None
        self.line_scatter_true_labels = None

        self.line_global_hamming_score = None
        self.line_partial_hamming_score = None

        self.line_global_hamming_loss = None
        self.line_partial_hamming_loss = None

        self.line_global_exact_match = None
        self.line_partial_exact_match = None

        self.line_global_j_index = None
        self.line_partial_j_index = None

        self.line_global_mse = None
        self.line_partial_mse = None

        self.line_global_mae = None
        self.line_partial_mae = None

        self.line_regression_true = None
        self.line_regression_pred = None

        #show configs
        self.num_plots = 0

        #subplot default
        self.subplot_performance = None
        self.subplot_kappa = None
        self.subplot_kappa_t = None
        self.subplot_kappa_m = None
        self.subplot_scatter_points = None
        self.subplot_hamming_score = None
        self.subplot_hamming_loss = None
        self.subplot_exact_match = None
        self.subplot_j_index = None
        self.subplot_mse = None
        self.subplot_mae = None
        self.subplot_true_vs_predicts = None

        if plots is not None:
            if len(plots) < 1:
                raise ValueError('No plots were given.')
            else:
                self.configure(n_wait, dataset_name, plots, n_learners)
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


    def configure(self, n_wait, dataset_name, plots, n_learners):
        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        warnings.filterwarnings("ignore", ".*left==right.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.plots = plots
        self.n_learners = n_learners
        self.X = []

        plt.ion()
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle(dataset_name)
        self.num_plots = len(self.plots)
        base = 11 + self.num_plots * 100
        self.fig.canvas.set_window_title('scikit-multiflow')

        if 'performance' in self.plots:
            self.partial_performance = [[] for i in range(self.n_learners)]
            self.global_performance = [[] for i in range(self.n_learners)]

            self.subplot_performance = self.fig.add_subplot(base)
            self.subplot_performance.set_title('Classifier\'s accuracy')
            self.subplot_performance.set_ylabel('Performance ratio')
            self.subplot_performance.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_performance = [None for i in range(self.n_learners)]
            self.line_global_performance = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_performance[i], = self.subplot_performance.plot(self.X, self.partial_performance[i],
                                                                               label='Classifier '+str(i)+' - Partial performance (last ' + str(
                                                                                   self.n_wait) + ' samples)', linestyle='--')
                self.line_global_performance[i], = self.subplot_performance.plot(self.X, self.global_performance[i],
                                                                              label='Classifier '+str(i)+' - Global performance')
                handle.append(self.line_partial_performance[i])
                handle.append(self.line_global_performance[i])

            self.subplot_performance.legend(handles=handle)
            self.subplot_performance.set_ylim([0, 1])


        if 'kappa' in self.plots:
            self.partial_kappa = [[] for i in range(self.n_learners)]
            self.global_kappa = [[] for i in range(self.n_learners)]

            self.subplot_kappa = self.fig.add_subplot(base)
            self.subplot_kappa.set_title('Classifier\'s Kappa')
            self.subplot_kappa.set_ylabel('Kappa statistic')
            self.subplot_kappa.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_kappa = [None for i in range(self.n_learners)]
            self.line_global_kappa = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa[i], = self.subplot_kappa.plot(self.X, self.partial_kappa[i],
                                                                   label='Classifier '+str(i)+' - Sliding window Kappa (last '
                                                                         + str(self.n_wait) + ' samples)')
                self.line_global_kappa[i], = self.subplot_kappa.plot(self.X, self.global_kappa[i], label='Classifier '+str(i)+' - Global kappa')
                handle.append(self.line_partial_kappa[i])
                handle.append(self.line_global_kappa[i])

            self.subplot_kappa.legend(handles=handle)
            self.subplot_kappa.set_ylim([-1, 1])

        if 'kappa_t' in self.plots:
            self.partial_kappa_t = [[] for i in range(self.n_learners)]
            self.global_kappa_t = [[] for i in range(self.n_learners)]

            self.subplot_kappa_t = self.fig.add_subplot(base)
            self.subplot_kappa_t.set_title('Classifier\'s Kappa T')
            self.subplot_kappa_t.set_ylabel('Kappa T statistic')
            self.subplot_kappa_t.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_kappa_t = [None for i in range(self.n_learners)]
            self.line_global_kappa_t = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa_t[i], = self.subplot_kappa_t.plot(self.X, self.partial_kappa_t[i],
                                                                       label='Classifier '+str(i)+' - Sliding window Kappa T (last '
                                                                             + str(self.n_wait) + ' samples)', linestyle='--')
                self.line_global_kappa_t[i], = self.subplot_kappa_t.plot(self.X, self.global_kappa_t[i], label='Classifier '+str(i)+' - Global kappa T')
                handle.append(self.line_partial_kappa_t[i])
                handle.append(self.line_global_kappa_t[i])

            self.subplot_kappa_t.legend(handles=handle)
            self.subplot_kappa_t.set_ylim([-1, 1])

        if 'kappa_m' in self.plots:
            self.partial_kappa_m = [[] for i in range(self.n_learners)]
            self.global_kappa_m = [[] for i in range(self.n_learners)]

            self.subplot_kappa_m = self.fig.add_subplot(base)
            self.subplot_kappa_m.set_title('Classifier\'s Kappa M')
            self.subplot_kappa_m.set_ylabel('Kappa M statistic')
            self.subplot_kappa_m.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_kappa_m = [None for i in range(self.n_learners)]
            self.line_global_kappa_m = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa_m[i], = self.subplot_kappa_m.plot(self.X, self.partial_kappa_m[i],
                                                                       label='Classifier '+str(i)+' - Sliding window kappa M (last '
                                                                             + str(self.n_wait) + ' samples)')
                self.line_global_kappa_m[i], = self.subplot_kappa_m.plot(self.X, self.global_kappa_m[i], label='Classifier '+str(i)+' - Global kappa M')
                handle.append(self.line_partial_kappa_m[i])
                handle.append(self.line_global_kappa_m[i])

            self.subplot_kappa_m.legend(handles=handle)
            self.subplot_kappa_m.set_ylim([-1, 1])

        if 'scatter' in self.plots:
            self.scatter_predicts = [[] for i in range(self.n_learners)]
            self.scatter_true_labels = []
            self.scatter_x = []
            self.subplot_scatter_points = self.fig.add_subplot(base)
            base += 1

            self.subplot_scatter_points.set_title('Predicts and true labels')
            self.subplot_scatter_points.set_ylabel('Class labels')
            self.subplot_scatter_points.set_xlabel('Sample analyzed')
            self.scatter_true_labels_colors = []
            self.scatter_predicts_colors = [[] for i in range(self.n_learners)]

        if 'hamming_score' in self.plots:
            self.global_hamming_score = [[] for i in range(self.n_learners)]
            self.partial_hamming_score = [[] for i in range(self.n_learners)]

            self.subplot_hamming_score = self.fig.add_subplot(base)
            self.subplot_hamming_score.set_title('Classifier\'s hamming score')
            self.subplot_hamming_score.set_ylabel('Hamming score')
            self.subplot_hamming_score.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_hamming_score = [None for i in range(self.n_learners)]
            self.line_global_hamming_score = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_hamming_score[i], = self.subplot_hamming_score.plot(self.X, self.partial_hamming_score[i],
                                                                                      label='Classifier '+str(i)+' - Partial Hamming score (last ' + str(
                                                                                   self.n_wait) + ' samples)')
                self.line_global_hamming_score[i], = self.subplot_hamming_score.plot(self.X, self.global_hamming_score[i],
                                                                              label='Classifier '+str(i)+' - Global Hamming score')
                handle.append(self.line_partial_hamming_score[i])
                handle.append(self.line_global_hamming_score[i])

            self.subplot_hamming_score.legend(handles=handle)
            self.subplot_hamming_score.set_ylim([0, 1])

        if 'hamming_loss' in self.plots:
            self.global_hamming_loss = [[] for i in range(self.n_learners)]
            self.partial_hamming_loss = [[] for i in range(self.n_learners)]

            self.subplot_hamming_loss = self.fig.add_subplot(base)
            self.subplot_hamming_loss.set_title('Classifier\'s hamming loss')
            self.subplot_hamming_loss.set_ylabel('Hamming loss')
            self.subplot_hamming_loss.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_hamming_loss = [None for i in range(self.n_learners)]
            self.line_global_hamming_loss = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_hamming_loss[i], = self.subplot_hamming_loss.plot(self.X, self.partial_hamming_loss[i],
                                                                               label='Classifier '+str(i)+' - Partial Hamming loss (last ' + str(
                                                                                   self.n_wait) + ' samples)')
                self.line_global_hamming_loss[i], = self.subplot_hamming_loss.plot(self.X, self.global_hamming_loss[i],
                                                                              label='Classifier '+str(i)+' - Global Hamming loss')
                handle.append(self.line_partial_hamming_loss[i])
                handle.append(self.line_global_hamming_loss[i])

            self.subplot_hamming_loss.legend(handles=handle)
            self.subplot_hamming_loss.set_ylim([0, 1])

        if 'exact_match' in self.plots:
            self.global_exact_match = [[] for i in range(self.n_learners)]
            self.partial_exact_match = [[] for i in range(self.n_learners)]

            self.subplot_exact_match = self.fig.add_subplot(base)
            self.subplot_exact_match.set_title('Classifier\'s exact matches')
            self.subplot_exact_match.set_ylabel('Exact matches')
            self.subplot_exact_match.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_exact_match = [None for i in range(self.n_learners)]
            self.line_global_exact_match = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_exact_match[i], = self.subplot_exact_match.plot(self.X, self.partial_exact_match[i],
                                                                               label='Classifier '+str(i)+' - Partial exact matches (last ' + str(
                                                                                   self.n_wait) + ' samples)')
                self.line_global_exact_match[i], = self.subplot_exact_match.plot(self.X, self.global_exact_match[i],
                                                                              label='Classifier '+str(i)+' - Global exact matches')
                handle.append(self.line_partial_exact_match[i])
                handle.append(self.line_global_exact_match[i])

            self.subplot_exact_match.legend(handles=handle)
            self.subplot_exact_match.set_ylim([0, 1])

        if 'j_index' in self.plots:
            self.global_j_index = [[] for i in range(self.n_learners)]
            self.partial_j_index = [[] for i in range(self.n_learners)]

            self.subplot_j_index = self.fig.add_subplot(base)
            self.subplot_j_index.set_title('Classifier\'s J index')
            self.subplot_j_index.set_ylabel('J index')
            self.subplot_j_index.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_j_index = [None for i in range(self.n_learners)]
            self.line_global_j_index = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_j_index[i], = self.subplot_j_index.plot(self.X, self.partial_j_index[i],
                                                                               label='Classifier '+str(i)+' - Partial j index (last ' + str(
                                                                                   self.n_wait) + ' samples)')
                self.line_global_j_index[i], = self.subplot_j_index.plot(self.X, self.global_j_index[i],
                                                                              label='Classifier '+str(i)+' - Global j index')
                handle.append(self.line_partial_j_index[i])
                handle.append(self.line_global_j_index[i])

            self.subplot_j_index.legend(handles=handle)
            self.subplot_j_index.set_ylim([0, 1])

        if 'mean_square_error' in self.plots:
            self.global_mse = [[] for i in range(self.n_learners)]
            self.partial_mse = [[] for i in range(self.n_learners)]

            self.subplot_mse = self.fig.add_subplot(base)
            self.subplot_mse.set_title('Regressor\'s MSE')
            self.subplot_mse.set_ylabel('MSE')
            self.subplot_mse.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_mse = [None for i in range(self.n_learners)]
            self.line_global_mse = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_mse[i], = self.subplot_mse.plot(self.X, self.partial_mse[i],
                                                                       label='Classifier '+str(i)+' - Partial MSE (last ' + str(
                                                                           self.n_wait) + ' samples)')
                self.line_global_mse[i], = self.subplot_mse.plot(self.X, self.global_mse[i],
                                                                      label='Classifier '+str(i)+' - Global MSE')
                handle.append(self.line_partial_mse[i])
                handle.append(self.line_global_mse[i])

            self.subplot_mse.legend(handles=handle)
            self.subplot_mse.set_ylim([0, 1])

        if 'mean_absolute_error' in self.plots:
            self.global_mae = [[] for i in range(self.n_learners)]
            self.partial_mae = [[] for i in range(self.n_learners)]

            self.subplot_mae = self.fig.add_subplot(base)
            self.subplot_mae.set_title('Regressor\'s MAE')
            self.subplot_mae.set_ylabel('MAE')
            self.subplot_mae.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_mae = [None for i in range(self.n_learners)]
            self.line_global_mae = [None for i in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_mae[i], = self.subplot_mae.plot(self.X, self.partial_mae[i],
                                                               label='Classifier '+str(i)+' - Partial MAE (last ' + str(
                                                                   self.n_wait) + ' samples)')
                self.line_global_mae[i], = self.subplot_mae.plot(self.X, self.global_mae[i],
                                                               label='Classifier '+str(i)+' - Global MAE')
                handle.append(self.line_partial_mae[i])
                handle.append(self.line_global_mae[i])

            self.subplot_mae.legend(handles=handle)
            self.subplot_mae.set_ylim([0, 1])

        if 'true_vs_predicts' in self.plots:
            self.regression_true = []
            self.regression_pred = [[] for i in range(self.n_learners)]

            self.subplot_true_vs_predicts = self.fig.add_subplot(base)
            self.subplot_true_vs_predicts.set_title('Regressor\'s True Labels vs Predictions')
            self.subplot_true_vs_predicts.set_ylabel('y')
            self.subplot_true_vs_predicts.set_xlabel('Samples analyzed')
            base += 1

            self.line_regression_true, = self.subplot_true_vs_predicts.plot(self.X, self.regression_true,
                                                                            label='True y')
            handle = [self.line_regression_true]

            for i in range(self.n_learners):
                self.line_regression_pred[i], = self.subplot_true_vs_predicts.plot(self.X, self.regression_pred[i],
                                                                                label='Classifier '+str(i)+' - Predicted y', linestyle='dotted')
                handle.append(self.line_regression_pred[i])

            self.subplot_true_vs_predicts.legend(handles=handle)
            self.subplot_true_vs_predicts.set_ylim([0,1])

        self.fig.subplots_adjust(hspace=.5)

        self.fig.tight_layout(pad=2.6, w_pad=0.5, h_pad=1.0)

    def draw(self, train_step, dict):
        self.X.append(train_step)

        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []

        if 'performance' in self.plots:
            for i in range(self.n_learners):
                self.global_performance[i].append(dict['performance'][i][0])
                self.partial_performance[i].append(dict['performance'][i][1])
                self.line_global_performance[i].set_data(self.X, self.global_performance[i])
                self.line_partial_performance[i].set_data(self.X, self.partial_performance[i])

                self.temp.append(self.subplot_performance.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['performance'][i][0], 3)),
                                                               xy=(train_step, dict['performance'][i][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
                self.temp.append(self.subplot_performance.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['performance'][i][1], 3)),
                                                               xy=(train_step, dict['performance'][i][1]),
                                                               xytext=(8, 0), textcoords='offset points'))

            self.subplot_performance.set_ylim([0, 1])
            self.subplot_performance.set_xlim([0, 1.2 * np.max(self.X)])


        if 'kappa' in self.plots:
            for i in range(self.n_learners):
                self.global_kappa[i].append(dict['kappa'][i][0])
                self.partial_kappa[i].append(dict['kappa'][i][1])
                self.line_global_kappa[i].set_data(self.X, self.global_kappa[i])
                self.line_partial_kappa[i].set_data(self.X, self.partial_kappa[i])

                self.temp.append(self.subplot_kappa.annotate('Clf '+str(i)+' - Global kappa: ' + str(round(dict['kappa'][i][0], 3)),
                                                             xy=(train_step, dict['kappa'][i][0]), xytext=(8, 0),
                                                             textcoords='offset points'))
                self.temp.append(self.subplot_kappa.annotate('Clf '+str(i)+' - Sliding window Kappa: ' + str(round(dict['kappa'][i][1], 3)),
                                                             xy=(train_step, dict['kappa'][i][1]), xytext=(8, 0),
                                                             textcoords='offset points'))

            self.subplot_kappa.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa.set_ylim([-1, 1])


        if 'kappa_t' in self.plots:
            minimum = -1.
            for i in range(self.n_learners):
                self.global_kappa_t[i].append(dict['kappa_t'][i][0])
                self.partial_kappa_t[i].append(dict['kappa_t'][i][1])
                self.line_global_kappa_t[i].set_data(self.X, self.global_kappa_t[i])
                self.line_partial_kappa_t[i].set_data(self.X, self.partial_kappa_t[i])

                self.temp.append(self.subplot_kappa_t.annotate('Clf '+str(i)+' - Global Kappa T: ' + str(round(dict['kappa_t'][i][0], 3)),
                                                             xy=(train_step, dict['kappa_t'][i][0]), xytext=(8, 0),
                                                             textcoords='offset points'))
                self.temp.append(self.subplot_kappa_t.annotate('Clf '+str(i)+' - Sliding window Kappa T: ' + str(round(dict['kappa_t'][i][1], 3)),
                                                             xy=(train_step, dict['kappa_t'][i][1]), xytext=(8, 0),
                                                             textcoords='offset points'))
                minimum = min(min(minimum, min(self.global_kappa_t[i])), min(minimum, min(self.partial_kappa_t[i])))

            self.subplot_kappa_t.set_xlim([0, 1.2 * np.max(self.X)])
            # self.subplot_kappa_t.set_ylim([min([min(self.global_kappa_t), min(self.partial_kappa_t), -1.]), 1.])
            self.subplot_kappa_t.set_ylim([minimum, 1.])


        if 'kappa_m' in self.plots:
            minimum = -1.
            for i in range(self.n_learners):
                self.global_kappa_m[i].append(dict['kappa_m'][i][0])
                self.partial_kappa_m[i].append(dict['kappa_m'][i][1])
                self.line_global_kappa_m[i].set_data(self.X, self.global_kappa_m[i])
                self.line_partial_kappa_m[i].set_data(self.X, self.partial_kappa_m[i])

                self.temp.append(self.subplot_kappa_m.annotate('Clf '+str(i)+' - Global kappa M: ' + str(round(dict['kappa_m'][i][0], 3)),
                                                               xy=(train_step, dict['kappa_m'][i][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
                self.temp.append(self.subplot_kappa_m.annotate('Clf '+str(i)+' - Sliding window Kappa M: ' + str(round(dict['kappa_m'][i][1], 3)),
                                                               xy=(train_step, dict['kappa_m'][i][1]), xytext=(8, 0),
                                                               textcoords='offset points'))
                minimum = min(min(minimum, min(self.global_kappa_m[i])), min(minimum, min(self.partial_kappa_m[i])))

            self.subplot_kappa_m.set_xlim([0, 1.2 * np.max(self.X)])
            # self.subplot_kappa_m.set_ylim([min([min(self.global_kappa_m), min(self.partial_kappa_m), -1.]), 1.])
            self.subplot_kappa_m.set_ylim([minimum, 1.])

        if 'scatter' in self.plots:
            self.scatter_x.append(train_step)
            self.scatter_true_labels.append(dict['scatter'][0][0])

            for i in range(self.n_learners):
                self.scatter_predicts[i].append(dict['scatter'][i][1])
                if dict['scatter'][0][0] == dict['scatter'][i][1]:
                    self.scatter_predicts_colors[i].append(self.COLOR_MAP[i%len(self.COLOR_MAP)])
                else:
                    self.scatter_predicts_colors[i].append(self.COLOR_MAP[i%len(self.COLOR_MAP)])

            self.scatter_true_labels_colors.append('g')

            aux_one = [item for sublist in self.scatter_predicts for item in sublist]
            classes = list(set().union(aux_one, self.scatter_true_labels))
            self.subplot_scatter_points.set_xlim(np.min(self.scatter_x), 1.2 * np.max(self.scatter_x))
            self.subplot_scatter_points.set_ylim(np.min(classes) - 1, np.max(classes) + 1)

            scat_true = self.subplot_scatter_points.scatter(self.scatter_x, self.scatter_true_labels, s=6,
                                                                label='True labels', c=self.scatter_true_labels_colors)
            for i in range(self.n_learners):
                scat_pred = self.subplot_scatter_points.scatter(self.scatter_x, self.scatter_predicts[i], s=6,
                                                                label='Predicts', c=self.scatter_predicts_colors[i])

            legend = []
            colour = []
            for i in range(self.n_learners):
                colour.append(self.COLOR_MAP[i%len(self.COLOR_MAP)])
                legend.append('Clf '+str(i)+' Prediction')
            colour.append('g')
            legend.append('True label')
            recs = []
            for i in range(0, len(colour)):
                recs.append(mpatches.Circle((0, 0), 1, fc=colour[i]))
            #self.subplot_scatter_points.legend(handles=[scat_true, scat_pred])
            self.subplot_scatter_points.legend(recs, legend, loc=4)

        if 'hamming_score' in self.plots:
            for i in range(self.n_learners):
                self.global_hamming_score[i].append(dict['hamming_score'][i][0])
                self.partial_hamming_score[i].append(dict['hamming_score'][i][1])
                self.line_global_hamming_score[i].set_data(self.X, self.global_hamming_score[i])
                self.line_partial_hamming_score[i].set_data(self.X, self.partial_hamming_score[i])

                self.temp.append(self.subplot_hamming_score.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['hamming_score'][i][0], 3)),
                                                                   xy=(train_step, dict['hamming_score'][i][0]), xytext=(8, 0),
                                                                   textcoords='offset points'))
                self.temp.append(self.subplot_hamming_score.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['hamming_score'][i][1], 3)),
                                                                   xy=(train_step, dict['hamming_score'][i][1]),
                                                                   xytext=(8, 0), textcoords='offset points'))

            self.subplot_hamming_score.set_ylim([0, 1])
            self.subplot_hamming_score.set_xlim([0, 1.2 * np.max(self.X)])

        if 'hamming_loss' in self.plots:
            for i in range(self.n_learners):
                self.global_hamming_loss[i].append(dict['hamming_loss'][i][0])
                self.partial_hamming_loss[i].append(dict['hamming_loss'][i][1])
                self.line_global_hamming_loss[i].set_data(self.X, self.global_hamming_loss[i])
                self.line_partial_hamming_loss[i].set_data(self.X, self.partial_hamming_loss[i])

                self.temp.append(self.subplot_hamming_loss.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['hamming_loss'][i][0], 3)),
                                                                   xy=(train_step, dict['hamming_loss'][i][0]), xytext=(8, 0),
                                                                   textcoords='offset points'))
                self.temp.append(self.subplot_hamming_loss.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['hamming_loss'][i][1], 3)),
                                                                   xy=(train_step, dict['hamming_loss'][i][1]),
                                                                   xytext=(8, 0), textcoords='offset points'))

            self.subplot_hamming_loss.set_ylim([0, 1])
            self.subplot_hamming_loss.set_xlim([0, 1.2 * np.max(self.X)])

        if 'exact_match' in self.plots:
            for i in range(self.n_learners):
                self.global_exact_match[i].append(dict['exact_match'][i][0])
                self.partial_exact_match[i].append(dict['exact_match'][i][1])
                self.line_global_exact_match[i].set_data(self.X, self.global_exact_match[i])
                self.line_partial_exact_match[i].set_data(self.X, self.partial_exact_match[i])

                self.temp.append(self.subplot_exact_match.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['exact_match'][i][0], 3)),
                                                                   xy=(train_step, dict['exact_match'][i][0]), xytext=(8, 0),
                                                                   textcoords='offset points'))
                self.temp.append(self.subplot_exact_match.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['exact_match'][i][1], 3)),
                                                                   xy=(train_step, dict['exact_match'][i][1]),
                                                                   xytext=(8, 0), textcoords='offset points'))

            self.subplot_exact_match.set_ylim([0, 1])
            self.subplot_exact_match.set_xlim([0, 1.2 * np.max(self.X)])

        if 'j_index' in self.plots:
            for i in range(self.n_learners):
                self.global_j_index[i].append(dict['j_index'][i][0])
                self.partial_j_index[i].append(dict['j_index'][i][1])
                self.line_global_j_index[i].set_data(self.X, self.global_j_index[i])
                self.line_partial_j_index[i].set_data(self.X, self.partial_j_index[i])

                self.temp.append(self.subplot_j_index.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['j_index'][i][0], 3)),
                                                                   xy=(train_step, dict['j_index'][i][0]), xytext=(8, 0),
                                                                   textcoords='offset points'))
                self.temp.append(self.subplot_j_index.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['j_index'][i][1], 3)),
                                                                   xy=(train_step, dict['j_index'][i][1]),
                                                                   xytext=(8, 0), textcoords='offset points'))

            self.subplot_j_index.set_ylim([0, 1])
            self.subplot_j_index.set_xlim([0, 1.2 * np.max(self.X)])

        if 'mean_square_error' in self.plots:
            minimum = 0
            maximum = 0
            for i in range(self.n_learners):
                self.global_mse[i].append(dict['mean_square_error'][i][0])
                self.partial_mse[i].append(dict['mean_square_error'][i][1])
                self.line_global_mse[i].set_data(self.X, self.global_mse[i])
                self.line_partial_mse[i].set_data(self.X, self.partial_mse[i])

                self.temp.append(self.subplot_mse.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['mean_square_error'][i][0], 6)),
                                                               xy=(train_step, dict['mean_square_error'][i][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
                self.temp.append(self.subplot_mse.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['mean_square_error'][i][1], 6)),
                                                               xy=(train_step, dict['mean_square_error'][i][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
                minimum = min([min(self.global_mse[i]), min(self.partial_mse[i]), minimum])
                maximum = max([max(self.global_mse[i]), max(self.partial_mse[i]), maximum])

            self.subplot_mse.set_ylim([minimum - 1.2*minimum, maximum + 1.2*maximum])
            self.subplot_mse.set_xlim([0, 1.2 * np.max(self.X)])

        if 'mean_absolute_error' in self.plots:
            minimum = 0
            maximum = 0
            for i in range(self.n_learners):
                self.global_mae[i].append(dict['mean_absolute_error'][i][0])
                self.partial_mae[i].append(dict['mean_absolute_error'][i][1])
                self.line_global_mae[i].set_data(self.X, self.global_mae[i])
                self.line_partial_mae[i].set_data(self.X, self.partial_mae[i])

                self.temp.append(self.subplot_mae.annotate('Clf '+str(i)+' - Global: ' + str(round(dict['mean_absolute_error'][i][0], 6)),
                                                           xy=(train_step, dict['mean_absolute_error'][i][0]), xytext=(8, 0),
                                                           textcoords='offset points'))
                self.temp.append(self.subplot_mae.annotate('Clf '+str(i)+' - Partial: ' + str(round(dict['mean_absolute_error'][i][1], 6)),
                                                           xy=(train_step, dict['mean_absolute_error'][i][1]),
                                                           xytext=(8, 0), textcoords='offset points'))
                minimum = min([min(self.global_mae[i]), min(self.partial_mae[i]), minimum])
                maximum = max([max(self.global_mae[i]), max(self.partial_mae[i]), maximum])

            self.subplot_mae.set_ylim([minimum - 1.2*minimum, maximum + 1.2*maximum])
            self.subplot_mae.set_xlim([0, 1.2 * np.max(self.X)])

        if 'true_vs_predicts' in self.plots:
            self.regression_true.append(dict['true_vs_predicts'][0][0])
            self.line_regression_true.set_data(self.X, self.regression_true)
            minimum = 0
            maximum = 0
            for i in range(self.n_learners):
                self.regression_pred[i].append(dict['true_vs_predicts'][i][1])
                #self.line_regression_true.set_data(self.X, self.regression_true)
                #self.line_regression_pred.set_data(self.X, self.regression_pred)
                scat_pred = self.subplot_true_vs_predicts.scatter(self.X, self.regression_pred[i], s=6,
                                                                label='Clf '+str(i)+' - Predictions', c=self.COLOR_MAP[i%len(self.COLOR_MAP)])
                minimum = min([min(self.regression_pred[i]), min(self.regression_true[i]), minimum])
                maximum = max([max(self.regression_pred[i]), max(self.regression_true[i]), maximum])

            self.subplot_true_vs_predicts.set_ylim([minimum - 1., maximum + 1.])
            self.subplot_true_vs_predicts.set_xlim([0, 1.2 * np.max(self.X)])

        plt.draw()
        plt.pause(0.0001)


    def draw_scatter_points(self, X, y, predict):
        pass

    def hold(self):
        plt.show(block=True)

    def get_info(self):
        pass

if __name__ == "__main__":
    ev = EvaluationVisualizer()
    print(ev.get_class_type())