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

            self.line_partial_kappa, = self.subplot_kappa.plot(self.X, self.partial_kappa,
                                                               label='Sliding window Kappa (last '
                                                                     + str(self.n_wait) + ' samples)')
            self.line_global_kappa, = self.subplot_kappa.plot(self.X, self.global_kappa, label='Global kappa')
            self.subplot_kappa.legend(handles=[self.line_partial_kappa, self.line_global_kappa])
            self.subplot_kappa.set_ylim([-1, 1])

        if 'kappa_t' in self.plots:
            self.partial_kappa_t = []
            self.global_kappa_t = []

            self.subplot_kappa_t = self.fig.add_subplot(base)
            self.subplot_kappa_t.set_title('Classifier\'s Kappa T')
            self.subplot_kappa_t.set_ylabel('Kappa T statistic')
            self.subplot_kappa_t.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_kappa_t, = self.subplot_kappa_t.plot(self.X, self.partial_kappa_t,
                                                                   label='Sliding window Kappa T (last '
                                                                         + str(self.n_wait) + ' samples)')
            self.line_global_kappa_t, = self.subplot_kappa_t.plot(self.X, self.global_kappa_t, label='Global kappa T')
            self.subplot_kappa_t.legend(handles=[self.line_partial_kappa_t, self.line_global_kappa_t])
            self.subplot_kappa_t.set_ylim([-1, 1])

        if 'kappa_m' in self.plots:
            self.partial_kappa_m = []
            self.global_kappa_m = []

            self.subplot_kappa_m = self.fig.add_subplot(base)
            self.subplot_kappa_m.set_title('Classifier\'s Kappa M')
            self.subplot_kappa_m.set_ylabel('Kappa M statistic')
            self.subplot_kappa_m.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_kappa_m, = self.subplot_kappa_m.plot(self.X, self.partial_kappa_m,
                                                                   label='Sliding window kappa M (last '
                                                                         + str(self.n_wait) + ' samples)')
            self.line_global_kappa_m, = self.subplot_kappa_m.plot(self.X, self.global_kappa_m, label='Global kappa M')
            self.subplot_kappa_m.legend(handles=[self.line_partial_kappa_m, self.line_global_kappa_m])
            self.subplot_kappa_m.set_ylim([-1, 1])

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

        if 'hamming_score' in self.plots:
            self.global_hamming_score = []
            self.partial_hamming_score = []

            self.subplot_hamming_score = self.fig.add_subplot(base)
            self.subplot_hamming_score.set_title('Classifier\'s hamming score')
            self.subplot_hamming_score.set_ylabel('Hamming score')
            self.subplot_hamming_score.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_hamming_score, = self.subplot_hamming_score.plot(self.X, self.partial_hamming_score,                                                         label='Partial Hamming score (last ' + str(
                                                                               self.n_wait) + ' samples)')
            self.line_global_hamming_score, = self.subplot_hamming_score.plot(self.X, self.global_hamming_score,
                                                                          label='Global Hamming score')
            self.subplot_hamming_score.legend(handles=[self.line_partial_hamming_score, self.line_global_hamming_score])
            self.subplot_hamming_score.set_ylim([0, 1])

        if 'hamming_loss' in self.plots:
            self.global_hamming_loss = []
            self.partial_hamming_loss = []

            self.subplot_hamming_loss = self.fig.add_subplot(base)
            self.subplot_hamming_loss.set_title('Classifier\'s hamming loss')
            self.subplot_hamming_loss.set_ylabel('Hamming loss')
            self.subplot_hamming_loss.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_hamming_loss, = self.subplot_hamming_loss.plot(self.X, self.partial_hamming_loss,
                                                                           label='Partial Hamming loss (last ' + str(
                                                                               self.n_wait) + ' samples)')
            self.line_global_hamming_loss, = self.subplot_hamming_loss.plot(self.X, self.global_hamming_loss,
                                                                          label='Global Hamming loss')
            self.subplot_hamming_loss.legend(handles=[self.line_partial_hamming_loss, self.line_global_hamming_loss])
            self.subplot_hamming_loss.set_ylim([0, 1])

        if 'exact_match' in self.plots:
            self.global_exact_match = []
            self.partial_exact_match = []

            self.subplot_exact_match = self.fig.add_subplot(base)
            self.subplot_exact_match.set_title('Classifier\'s exact matches')
            self.subplot_exact_match.set_ylabel('Exact matches')
            self.subplot_exact_match.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_exact_match, = self.subplot_exact_match.plot(self.X, self.partial_exact_match,
                                                                           label='Partial exact matches (last ' + str(
                                                                               self.n_wait) + ' samples)')
            self.line_global_exact_match, = self.subplot_exact_match.plot(self.X, self.global_exact_match,
                                                                          label='Global exact matches')
            self.subplot_exact_match.legend(handles=[self.line_partial_exact_match, self.line_global_exact_match])
            self.subplot_exact_match.set_ylim([0, 1])

        if 'j_index' in self.plots:
            self.global_j_index = []
            self.partial_j_index = []

            self.subplot_j_index = self.fig.add_subplot(base)
            self.subplot_j_index.set_title('Classifier\'s J index')
            self.subplot_j_index.set_ylabel('J index')
            self.subplot_j_index.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_j_index, = self.subplot_j_index.plot(self.X, self.partial_j_index,
                                                                           label='Partial j index (last ' + str(
                                                                               self.n_wait) + ' samples)')
            self.line_global_j_index, = self.subplot_j_index.plot(self.X, self.global_j_index,
                                                                          label='Global j index')
            self.subplot_j_index.legend(handles=[self.line_partial_j_index, self.line_global_j_index])
            self.subplot_j_index.set_ylim([0, 1])

        if 'mean_square_error' in self.plots:
            self.global_mse = []
            self.partial_mse = []

            self.subplot_mse = self.fig.add_subplot(base)
            self.subplot_mse.set_title('Classifier\'s MSE')
            self.subplot_mse.set_ylabel('MSE')
            self.subplot_mse.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_mse, = self.subplot_mse.plot(self.X, self.partial_mse,
                                                                   label='Partial MSE (last ' + str(
                                                                       self.n_wait) + ' samples)')
            self.line_global_mse, = self.subplot_mse.plot(self.X, self.global_mse,
                                                                  label='Global MSE')
            self.subplot_mse.legend(handles=[self.line_partial_mse, self.line_global_mse])
            self.subplot_mse.set_ylim([0, 1])

        if 'mean_absolute_error' in self.plots:
            self.global_mae = []
            self.partial_mae = []

            self.subplot_mae = self.fig.add_subplot(base)
            self.subplot_mae.set_title('Classifier\'s MAE')
            self.subplot_mae.set_ylabel('MAE')
            self.subplot_mae.set_xlabel('Samples analyzed')
            base += 1

            self.line_partial_mae, = self.subplot_mae.plot(self.X, self.partial_mae,
                                                           label='Partial MAE (last ' + str(
                                                               self.n_wait) + ' samples)')
            self.line_global_mae, = self.subplot_mae.plot(self.X, self.global_mae,
                                                           label='Global MAE')
            self.subplot_mae.legend(handles=[self.line_partial_mae, self.line_global_mae])
            self.subplot_mae.set_ylim([0, 1])

        if 'true_vs_predicts' in self.plots:
            self.regression_true = []
            self.regression_pred = []

            self.subplot_true_vs_predicts = self.fig.add_subplot(base)
            self.subplot_true_vs_predicts.set_title('Classifier\'s True Labels vs Predictions')
            self.subplot_true_vs_predicts.set_ylabel('y')
            self.subplot_true_vs_predicts.set_xlabel('Samples analyzed')
            base += 1

            self.line_regression_true, = self.subplot_true_vs_predicts.plot(self.X, self.regression_true,
                                                                            label='True y')
            self.line_regression_pred, = self.subplot_true_vs_predicts.plot(self.X, self.regression_pred,
                                                                            label='Predicted y', linestyle='dotted')
            self.subplot_true_vs_predicts.legend(handles=[self.line_regression_true, self.line_regression_pred])
            self.subplot_true_vs_predicts.set_ylim([0,1])

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


        if 'kappa_t' in self.plots:
            self.global_kappa_t.append(dict['kappa_t'][0])
            self.partial_kappa_t.append(dict['kappa_t'][1])
            self.line_global_kappa_t.set_data(self.X, self.global_kappa_t)
            self.line_partial_kappa_t.set_data(self.X, self.partial_kappa_t)

            self.temp.append(self.subplot_kappa_t.annotate('Global Kappa T: ' + str(round(dict['kappa_t'][0], 3)),
                                                         xy=(train_step, dict['kappa_t'][0]), xytext=(8, 0),
                                                         textcoords='offset points'))
            self.temp.append(self.subplot_kappa_t.annotate('Sliding window Kappa T: ' + str(round(dict['kappa_t'][1], 3)),
                                                         xy=(train_step, dict['kappa_t'][1]), xytext=(8, 0),
                                                         textcoords='offset points'))
            self.subplot_kappa_t.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa_t.set_ylim([min([min(self.global_kappa_t), min(self.partial_kappa_t), -1.]), 1.])


        if 'kappa_m' in self.plots:
            self.global_kappa_m.append(dict['kappa_m'][0])
            self.partial_kappa_m.append(dict['kappa_m'][1])
            self.line_global_kappa_m.set_data(self.X, self.global_kappa_m)
            self.line_partial_kappa_m.set_data(self.X, self.partial_kappa_m)

            self.temp.append(self.subplot_kappa_m.annotate('Global kappa M: ' + str(round(dict['kappa_m'][0], 3)),
                                                           xy=(train_step, dict['kappa_m'][0]), xytext=(8, 0),
                                                           textcoords='offset points'))
            self.temp.append(self.subplot_kappa_m.annotate('Sliding window Kappa M: ' + str(round(dict['kappa_m'][1], 3)),
                                                           xy=(train_step, dict['kappa_m'][1]), xytext=(8, 0),
                                                           textcoords='offset points'))
            self.subplot_kappa_m.set_xlim([0, 1.2 * np.max(self.X)])
            self.subplot_kappa_m.set_ylim([min([min(self.global_kappa_m), min(self.partial_kappa_m), -1.]), 1.])


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

        if 'hamming_score' in self.plots:
            self.global_hamming_score.append(dict['hamming_score'][0])
            self.partial_hamming_score.append(dict['hamming_score'][1])
            self.line_global_hamming_score.set_data(self.X, self.global_hamming_score)
            self.line_partial_hamming_score.set_data(self.X, self.partial_hamming_score)

            self.temp.append(self.subplot_hamming_score.annotate('Global: ' + str(round(dict['hamming_score'][0], 3)),
                                                               xy=(train_step, dict['hamming_score'][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
            self.temp.append(self.subplot_hamming_score.annotate('Partial: ' + str(round(dict['hamming_score'][1], 3)),
                                                               xy=(train_step, dict['hamming_score'][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_hamming_score.set_ylim([0, 1])
            self.subplot_hamming_score.set_xlim([0, 1.2 * np.max(self.X)])

        if 'hamming_loss' in self.plots:
            self.global_hamming_loss.append(dict['hamming_loss'][0])
            self.partial_hamming_loss.append(dict['hamming_loss'][1])
            self.line_global_hamming_loss.set_data(self.X, self.global_hamming_loss)
            self.line_partial_hamming_loss.set_data(self.X, self.partial_hamming_loss)

            self.temp.append(self.subplot_hamming_loss.annotate('Global: ' + str(round(dict['hamming_loss'][0], 3)),
                                                               xy=(train_step, dict['hamming_loss'][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
            self.temp.append(self.subplot_hamming_loss.annotate('Partial: ' + str(round(dict['hamming_loss'][1], 3)),
                                                               xy=(train_step, dict['hamming_loss'][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_hamming_loss.set_ylim([0, 1])
            self.subplot_hamming_loss.set_xlim([0, 1.2 * np.max(self.X)])

        if 'exact_match' in self.plots:
            self.global_exact_match.append(dict['exact_match'][0])
            self.partial_exact_match.append(dict['exact_match'][1])
            self.line_global_exact_match.set_data(self.X, self.global_exact_match)
            self.line_partial_exact_match.set_data(self.X, self.partial_exact_match)

            self.temp.append(self.subplot_exact_match.annotate('Global: ' + str(round(dict['exact_match'][0], 3)),
                                                               xy=(train_step, dict['exact_match'][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
            self.temp.append(self.subplot_exact_match.annotate('Partial: ' + str(round(dict['exact_match'][1], 3)),
                                                               xy=(train_step, dict['exact_match'][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_exact_match.set_ylim([0, 1])
            self.subplot_exact_match.set_xlim([0, 1.2 * np.max(self.X)])

        if 'j_index' in self.plots:
            self.global_j_index.append(dict['j_index'][0])
            self.partial_j_index.append(dict['j_index'][1])
            self.line_global_j_index.set_data(self.X, self.global_j_index)
            self.line_partial_j_index.set_data(self.X, self.partial_j_index)

            self.temp.append(self.subplot_j_index.annotate('Global: ' + str(round(dict['j_index'][0], 3)),
                                                               xy=(train_step, dict['j_index'][0]), xytext=(8, 0),
                                                               textcoords='offset points'))
            self.temp.append(self.subplot_j_index.annotate('Partial: ' + str(round(dict['j_index'][1], 3)),
                                                               xy=(train_step, dict['j_index'][1]),
                                                               xytext=(8, 0), textcoords='offset points'))
            self.subplot_j_index.set_ylim([0, 1])
            self.subplot_j_index.set_xlim([0, 1.2 * np.max(self.X)])

        if 'mean_square_error' in self.plots:
            self.global_mse.append(dict['mean_square_error'][0])
            self.partial_mse.append(dict['mean_square_error'][1])
            self.line_global_mse.set_data(self.X, self.global_mse)
            self.line_partial_mse.set_data(self.X, self.partial_mse)

            self.temp.append(self.subplot_mse.annotate('Global: ' + str(round(dict['mean_square_error'][0], 6)),
                                                           xy=(train_step, dict['mean_square_error'][0]), xytext=(8, 0),
                                                           textcoords='offset points'))
            self.temp.append(self.subplot_mse.annotate('Partial: ' + str(round(dict['mean_square_error'][1], 6)),
                                                           xy=(train_step, dict['mean_square_error'][1]),
                                                           xytext=(8, 0), textcoords='offset points'))
            self.subplot_mse.set_ylim([min([min(self.global_mse), min(self.partial_mse)]) - 1.2*min([min(self.global_mse), min(self.partial_mse)]),
                                       max([max(self.global_mse), max(self.partial_mse)]) + 1.2*max([max(self.global_mse), max(self.partial_mse)])])
            self.subplot_mse.set_xlim([0, 1.2 * np.max(self.X)])

        if 'mean_absolute_error' in self.plots:
            self.global_mae.append(dict['mean_absolute_error'][0])
            self.partial_mae.append(dict['mean_absolute_error'][1])
            self.line_global_mae.set_data(self.X, self.global_mae)
            self.line_partial_mae.set_data(self.X, self.partial_mae)

            self.temp.append(self.subplot_mae.annotate('Global: ' + str(round(dict['mean_absolute_error'][0], 6)),
                                                       xy=(train_step, dict['mean_absolute_error'][0]), xytext=(8, 0),
                                                       textcoords='offset points'))
            self.temp.append(self.subplot_mae.annotate('Partial: ' + str(round(dict['mean_absolute_error'][1], 6)),
                                                       xy=(train_step, dict['mean_absolute_error'][1]),
                                                       xytext=(8, 0), textcoords='offset points'))
            self.subplot_mae.set_ylim([min([min(self.global_mae), min(self.partial_mae)]) - 1.2*min([min(self.global_mae), min(self.partial_mae)]),
                                       max([max(self.global_mae), max(self.partial_mae)]) + 1.2*max([max(self.global_mae), max(self.partial_mae)])])
            self.subplot_mae.set_xlim([0, 1.2 * np.max(self.X)])

        if 'true_vs_predicts' in self.plots:
            self.regression_true.append(dict['true_vs_predicts'][0])
            self.regression_pred.append(dict['true_vs_predicts'][1])
            self.line_regression_true.set_data(self.X, self.regression_true)
            self.line_regression_pred.set_data(self.X, self.regression_pred)

            self.subplot_true_vs_predicts.set_ylim([min([min(self.regression_true), min(self.regression_pred)]) - 1.,
                                       max([max(self.regression_true), max(self.regression_pred)]) + 1.])
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