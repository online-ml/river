__author__ = 'Guilherme Matsumoto'

import warnings
from skmultiflow.visualization.base_listener import BaseListener
from matplotlib.rcsetup import cycler
import matplotlib.pyplot as plt


class EvaluationVisualizer(BaseListener):
    """ EvaluationVisualizer
    
    This class is responsible for maintaining and updating the plot modules 
    for all the evaluators in scikit-multiflow. 
    
    It uses matplotlib's pyplot modules to create the main plot, which 
    depending on the options passed to it as parameter, will create multiple 
    subplots to better display all requested metrics.
    
    The plots are updated on the go, at each n_wait samples. The plot is 
    redrawn at each step, which may cause significant slow down, depending on 
    the processor used and the plotting options.
    
    Line objects are used to describe performance measurements and scatter 
    instances will represent true labels and predictions, when requested.
    
    It supports the visualization of multiple learners per subplot as a way 
    of comparing the performance of different learning algorithms facing the 
    same data stream.
    
    Parameters
    ----------
    n_wait: int (Default: 200)
        The interval between each plot point.
    
    dataset_name: string (Default: 'Unnamed graph')
        The title of the plot. Algorithmically it's not important.
    
    plots: list
        A list containing all the subplots to plot. Can be any of: 
        'performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 
        'exact_match', 'j_index', 'mean_square_error', 'mean_absolute_error', 
        'true_vs_predicts', 'kappa_t', 'kappa_m'
    
    n_learners: int
        The number of learners to compare.
    
    Raises
    ------
    ValueError: A ValueError can be raised for a series of reasons. If no plots 
    are passed as parameter to the constructor a ValueError is raised. If the wrong 
    type of parameter is passed to on_new_train_step the same error is raised.
    
    Notes
    -----
    Using more than 3 plot types at a time is not recommended, as it will 
    significantly slow down processing. Also, for the same reason comparing 
    more than 3 learners at a time is not recommended.
    
    """

    def __init__(self, task_type=None, n_wait=200, dataset_name='Unnamed graph', plots=None, n_learners=1):
        super().__init__()

        # Default values
        self.X = None
        self.scatter_x = None
        self._is_legend_set = False
        self._draw_cnt = 0

        self.temp = []

        self.true_values = None
        self.pred_values = None

        self.partial_performance = None
        self.global_performance = None

        self.global_kappa = None
        self.partial_kappa = None

        self.global_kappa_t = None
        self.partial_kappa_t = None

        self.global_kappa_m = None
        self.partial_kappa_m = None

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

        # Configuration
        self.n_wait = None
        self.dataset_name = None
        self.n_learners = None
        self.num_plots = 0

        # Lines
        self.line_global_performance = None
        self.line_partial_performance = None

        self.line_global_kappa = None
        self.line_partial_kappa = None

        self.line_global_kappa_t = None
        self.line_partial_kappa_t = None

        self.line_global_kappa_m = None
        self.line_partial_kappa_m = None

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

        self.line_true = None
        self.line_pred = None

        # Subplot default
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

        if task_type is None:
            raise ValueError('Task type for visualizer object is undefined.')
        else:
            if task_type in ['classification', 'regression', 'multi_output']:
                self.task_type = task_type
            else:
                raise ValueError('Invalid task type: {}'.format(task_type))

        if plots is not None:
            if len(plots) < 1:
                raise ValueError('No plots were given.')
            else:
                self.__configure(n_wait, dataset_name, plots, n_learners)
        else:
            raise ValueError('No plots were given.')

    def on_new_train_step(self, train_step, metrics_dict):
        """ on_new_train_step
        
        This is the listener main function, which gives it the ability to 
        'listen' for the caller. Whenever the EvaluationVisualiser should 
        be aware of some new data, the caller will call this function, 
        passing the new data as parameter.
        
        Parameters
        ----------
        train_step: int
            The number of samples processed to this moment.
        
        metrics_dict: dictionary
            A dictionary containing metric measurements, where the key is
            the metric name and the value its corresponding measurement.
            
        Raises
        ------
        ValueError: If wrong data formats are passed as parameter this error 
        is raised.
         
        """
        if train_step % self.n_wait == 0:
            try:
                self.draw(train_step, metrics_dict)
            except BaseException as exc:
                raise ValueError('Failed when trying to draw plot. ', exc)

    def on_new_scatter_data(self, X, y, prediction):
        pass

    def __configure(self, n_wait, dataset_name, plots, n_learners):
        """ __configure
        
        This function will verify which subplots it should create. For each one 
        of those, it will initialize all relevant objects to keep track of the 
        plotting points.
        
        Basic structures needed to keep track of plot values (for each subplot) 
        are: lists of values and matplotlib's line objects.
        
        The __configure function will also initialize each subplot with the 
        correct name and setup the axis.
        
        The subplot size will self adjust to each screen size, so that data can 
        be better viewed in different contexts.
        
        Parameters
        ----------
        n_wait: int (Default: 200)
            The interval between each plot point.
    
        dataset_name: string (Default: 'Unnamed graph')
            The title of the plot. Algorithmically it's not important.
    
        plots: list
            A list containing all the subplots to plot. Can be any of: 
            'performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 
            'exact_match', 'j_index', 'mean_square_error', 'mean_absolute_error', 
            'true_vs_predicts', 'kappa_t', 'kappa_m'
        
        n_learners: int
            The number of learners to compare.
         
        """
        font_size_small = 8
        font_size_medium = 10
        font_size_large = 12

        plt.rc('font', size=font_size_small)  # controls default text sizes
        plt.rc('axes', titlesize=font_size_medium)  # font size of the axes title
        plt.rc('axes', labelsize=font_size_small)  # font size of the x and y labels
        plt.rc('xtick', labelsize=font_size_small)  # font size of the tick labels
        plt.rc('ytick', labelsize=font_size_small)  # font size of the tick labels
        plt.rc('legend', fontsize=font_size_small)  # legend font size
        plt.rc('figure', titlesize=font_size_large)  # font size of the figure title

        warnings.filterwarnings("ignore", ".*GUI is implemented.*")
        warnings.filterwarnings("ignore", ".*left==right.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.plots = plots
        self.n_learners = n_learners
        self.X = []

        plt.ion()
        self.fig = plt.figure(figsize=(9, 5))
        self.fig.suptitle(dataset_name)
        self.num_plots = len(self.plots)
        base = 11 + self.num_plots * 100  # 3-digit integer describing the position of the subplot.
        self.fig.canvas.set_window_title('scikit-multiflow')

        if 'performance' in self.plots:
            self.partial_performance = [[] for _ in range(self.n_learners)]
            self.global_performance = [[] for _ in range(self.n_learners)]

            self.subplot_performance = self.fig.add_subplot(base)
            self.subplot_performance.set_title('Accuracy')
            self.subplot_performance.set_ylabel('Performance ratio')
            base += 1

            self.line_partial_performance = [None for _ in range(self.n_learners)]
            self.line_global_performance = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_performance[i], = self.subplot_performance.plot(
                    self.X,
                    self.partial_performance[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_performance[i], = self.subplot_performance.plot(
                    self.X, self.global_performance[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_performance[i])
                handle.append(self.line_global_performance[i])

            self._set_fig_legend(handle)
            self.subplot_performance.set_ylim(0, 1)

        if 'kappa' in self.plots:
            self.partial_kappa = [[] for _ in range(self.n_learners)]
            self.global_kappa = [[] for _ in range(self.n_learners)]

            self.subplot_kappa = self.fig.add_subplot(base)
            self.subplot_kappa.set_title('Kappa')
            self.subplot_kappa.set_ylabel('Kappa statistic')
            base += 1

            self.line_partial_kappa = [None for _ in range(self.n_learners)]
            self.line_global_kappa = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa[i], = self.subplot_kappa.plot(
                    self.X, self.partial_kappa[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_kappa[i], = self.subplot_kappa.plot(
                    self.X, self.global_kappa[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_kappa[i])
                handle.append(self.line_global_kappa[i])

            self._set_fig_legend(handle)
            self.subplot_kappa.set_ylim(-1, 1)

        if 'kappa_t' in self.plots:
            self.partial_kappa_t = [[] for _ in range(self.n_learners)]
            self.global_kappa_t = [[] for _ in range(self.n_learners)]

            self.subplot_kappa_t = self.fig.add_subplot(base)
            self.subplot_kappa_t.set_title('Kappa T')
            self.subplot_kappa_t.set_ylabel('Kappa T statistic')
            base += 1

            self.line_partial_kappa_t = [None for _ in range(self.n_learners)]
            self.line_global_kappa_t = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa_t[i], = self.subplot_kappa_t.plot(
                    self.X, self.partial_kappa_t[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_kappa_t[i], = self.subplot_kappa_t.plot(
                    self.X, self.global_kappa_t[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_kappa_t[i])
                handle.append(self.line_global_kappa_t[i])

            self._set_fig_legend(handle)
            self.subplot_kappa_t.set_ylim(-1, 1)

        if 'kappa_m' in self.plots:
            self.partial_kappa_m = [[] for _ in range(self.n_learners)]
            self.global_kappa_m = [[] for _ in range(self.n_learners)]

            self.subplot_kappa_m = self.fig.add_subplot(base)
            self.subplot_kappa_m.set_title('Kappa M')
            self.subplot_kappa_m.set_ylabel('Kappa M statistic')
            base += 1

            self.line_partial_kappa_m = [None for _ in range(self.n_learners)]
            self.line_global_kappa_m = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_kappa_m[i], = self.subplot_kappa_m.plot(
                    self.X, self.partial_kappa_m[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_kappa_m[i], = self.subplot_kappa_m.plot(
                    self.X, self.global_kappa_m[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_kappa_m[i])
                handle.append(self.line_global_kappa_m[i])

            self._set_fig_legend(handle)
            self.subplot_kappa_m.set_ylim(-1, 1)

        if 'hamming_score' in self.plots:
            self.global_hamming_score = [[] for _ in range(self.n_learners)]
            self.partial_hamming_score = [[] for _ in range(self.n_learners)]

            self.subplot_hamming_score = self.fig.add_subplot(base)
            self.subplot_hamming_score.set_title('Hamming score')
            self.subplot_hamming_score.set_ylabel('Hamming score')
            base += 1

            self.line_partial_hamming_score = [None for _ in range(self.n_learners)]
            self.line_global_hamming_score = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_hamming_score[i], = self.subplot_hamming_score.plot(
                    self.X, self.partial_hamming_score[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_hamming_score[i], = self.subplot_hamming_score.plot(
                    self.X, self.global_hamming_score[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_hamming_score[i])
                handle.append(self.line_global_hamming_score[i])

            self._set_fig_legend(handle)
            self.subplot_hamming_score.set_ylim(0, 1)

        if 'hamming_loss' in self.plots:
            self.global_hamming_loss = [[] for _ in range(self.n_learners)]
            self.partial_hamming_loss = [[] for _ in range(self.n_learners)]

            self.subplot_hamming_loss = self.fig.add_subplot(base)
            self.subplot_hamming_loss.set_title('Hamming loss')
            self.subplot_hamming_loss.set_ylabel('Hamming loss')
            base += 1

            self.line_partial_hamming_loss = [None for _ in range(self.n_learners)]
            self.line_global_hamming_loss = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_hamming_loss[i], = self.subplot_hamming_loss.plot(
                    self.X, self.partial_hamming_loss[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_hamming_loss[i], = self.subplot_hamming_loss.plot(
                    self.X, self.global_hamming_loss[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_hamming_loss[i])
                handle.append(self.line_global_hamming_loss[i])

            self._set_fig_legend(handle)
            self.subplot_hamming_loss.set_ylim(0, 1)

        if 'exact_match' in self.plots:
            self.global_exact_match = [[] for _ in range(self.n_learners)]
            self.partial_exact_match = [[] for _ in range(self.n_learners)]

            self.subplot_exact_match = self.fig.add_subplot(base)
            self.subplot_exact_match.set_title('Exact matches')
            self.subplot_exact_match.set_ylabel('Exact matches')
            base += 1

            self.line_partial_exact_match = [None for _ in range(self.n_learners)]
            self.line_global_exact_match = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_exact_match[i], = self.subplot_exact_match.plot(
                    self.X, self.partial_exact_match[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_exact_match[i], = self.subplot_exact_match.plot(
                    self.X, self.global_exact_match[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_exact_match[i])
                handle.append(self.line_global_exact_match[i])

            self._set_fig_legend(handle)
            self.subplot_exact_match.set_ylim(0, 1)

        if 'j_index' in self.plots:
            self.global_j_index = [[] for _ in range(self.n_learners)]
            self.partial_j_index = [[] for _ in range(self.n_learners)]

            self.subplot_j_index = self.fig.add_subplot(base)
            self.subplot_j_index.set_title('J index')
            self.subplot_j_index.set_ylabel('J index')
            base += 1

            self.line_partial_j_index = [None for _ in range(self.n_learners)]
            self.line_global_j_index = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_j_index[i], = self.subplot_j_index.plot(
                    self.X, self.partial_j_index[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_j_index[i], = self.subplot_j_index.plot(
                    self.X, self.global_j_index[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_j_index[i])
                handle.append(self.line_global_j_index[i])

            self._set_fig_legend(handle)
            self.subplot_j_index.set_ylim(0, 1)

        if 'mean_square_error' in self.plots:
            self.global_mse = [[] for _ in range(self.n_learners)]
            self.partial_mse = [[] for _ in range(self.n_learners)]

            self.subplot_mse = self.fig.add_subplot(base)
            self.subplot_mse.set_title('Mean Squared Error')
            self.subplot_mse.set_ylabel('MSE')
            base += 1

            self.line_partial_mse = [None for _ in range(self.n_learners)]
            self.line_global_mse = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_mse[i], = self.subplot_mse.plot(
                    self.X, self.partial_mse[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_mse[i], = self.subplot_mse.plot(
                    self.X, self.global_mse[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_mse[i])
                handle.append(self.line_global_mse[i])

            self._set_fig_legend(handle)
            self.subplot_mse.set_ylim(0, 1)

        if 'mean_absolute_error' in self.plots:
            self.global_mae = [[] for _ in range(self.n_learners)]
            self.partial_mae = [[] for _ in range(self.n_learners)]

            self.subplot_mae = self.fig.add_subplot(base)
            self.subplot_mae.set_title('Mean Absolute Error')
            self.subplot_mae.set_ylabel('MAE')
            base += 1

            self.line_partial_mae = [None for _ in range(self.n_learners)]
            self.line_global_mae = [None for _ in range(self.n_learners)]
            handle = []

            for i in range(self.n_learners):
                self.line_partial_mae[i], = self.subplot_mae.plot(
                    self.X, self.partial_mae[i],
                    label='Learner {}  (last {} samples)'.format(i, self.n_wait))
                self.line_global_mae[i], = self.subplot_mae.plot(
                    self.X, self.global_mae[i],
                    label='Learner {}'.format(i), linestyle='dotted')
                handle.append(self.line_partial_mae[i])
                handle.append(self.line_global_mae[i])

            self._set_fig_legend(handle)
            self.subplot_mae.set_ylim(0, 1)

        if 'true_vs_predicts' in self.plots:
            self.true_values = []
            self.pred_values = [[] for _ in range(self.n_learners)]

            self.subplot_true_vs_predicts = self.fig.add_subplot(base)
            self.subplot_true_vs_predicts.set_title('True vs Predicted')
            self.subplot_true_vs_predicts.set_ylabel('y')
            self.subplot_true_vs_predicts.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))
            base += 1

            if self.task_type == 'classification':
                self.line_true, = self.subplot_true_vs_predicts.step(self.X, self.true_values,
                                                                     label='True value')
            else:
                self.line_true, = self.subplot_true_vs_predicts.plot(self.X, self.true_values,
                                                                     label='True value')
            handle = [self.line_true]

            self.line_pred = [None for _ in range(self.n_learners)]

            for i in range(self.n_learners):
                if self.task_type == 'classification':
                    self.line_pred[i], = self.subplot_true_vs_predicts.step(self.X, self.pred_values[i],
                                                                            label='Learner {}'.format(i),
                                                                            linestyle='dotted')
                else:
                    self.line_pred[i], = self.subplot_true_vs_predicts.plot(self.X, self.pred_values[i],
                                                                            label='Learner {}'.format(i),
                                                                            linestyle='dotted')
                handle.append(self.line_pred[i])

            self.subplot_true_vs_predicts.legend(handles=handle)
            self.subplot_true_vs_predicts.set_ylim(0, 1)

        plt.xlabel('Samples')
        self.fig.subplots_adjust(hspace=.5)
        self.fig.tight_layout(rect=[0, .04, 1, 0.98], pad=2.6, w_pad=0.5, h_pad=1.0)

    def _set_fig_legend(self, handles=None):
        if not self._is_legend_set:
            self.fig.legend(handles=handles, ncol=self.n_learners, bbox_to_anchor=(0.02, 0.0), loc="lower left")
            self._is_legend_set = True

    def draw(self, train_step, metrics_dict):
        """ draw
        
        Updates and redraws the plot.
        
        Parameters
        ----------
        train_step: int
            The number of samples processed to this moment.
        
        metrics_dict: dictionary
            A dictionary containing tuples, where the first element is the 
            string that identifies one of the plot's subplot names, and the 
            second element is its numerical value.
             
        """
        self.X.append(train_step)

        for i in range(len(self.temp)):
            self.temp[i].remove()
        self.temp = []

        xy_pos_default = (1.01, .90)  # Default xy position for metric annotations

        if 'performance' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_performance.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_performance[i].append(metrics_dict['performance'][i][0])
                self.partial_performance[i].append(metrics_dict['performance'][i][1])
                self.line_global_performance[i].set_data(self.X, self.global_performance[i])
                self.line_partial_performance[i].set_data(self.X, self.partial_performance[i])

                self.temp.append(self.subplot_performance.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                metrics_dict['performance'][i][0],
                                                                metrics_dict['performance'][i][1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y

            self.subplot_performance.set_xlim(0, self.X[-1])
            self.subplot_performance.set_ylim(0, 1)

        if 'kappa' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_kappa.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_kappa[i].append(metrics_dict['kappa'][i][0])
                self.partial_kappa[i].append(metrics_dict['kappa'][i][1])
                self.line_global_kappa[i].set_data(self.X, self.global_kappa[i])
                self.line_partial_kappa[i].set_data(self.X, self.partial_kappa[i])

                self.temp.append(self.subplot_kappa.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_kappa[i][-1],
                                                                self.partial_kappa[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points', annotation_clip=False))
                shift_y += shift_y

            self.subplot_kappa.set_xlim(0, self.X[-1])
            self.subplot_kappa.set_ylim(0, 1)

        if 'kappa_t' in self.plots:
            minimum = -1.
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_kappa_t.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_kappa_t[i].append(metrics_dict['kappa_t'][i][0])
                self.partial_kappa_t[i].append(metrics_dict['kappa_t'][i][1])
                self.line_global_kappa_t[i].set_data(self.X, self.global_kappa_t[i])
                self.line_partial_kappa_t[i].set_data(self.X, self.partial_kappa_t[i])

                self.temp.append(self.subplot_kappa_t.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_kappa_t[i][-1],
                                                                self.partial_kappa_t[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y
                minimum = min(min(minimum, min(self.global_kappa_t[i])), min(minimum, min(self.partial_kappa_t[i])))

            self.subplot_kappa_t.set_xlim(0, self.X[-1])
            self.subplot_kappa_t.set_ylim([minimum, 1.])

        if 'kappa_m' in self.plots:
            minimum = -1.
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_kappa_m.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_kappa_m[i].append(metrics_dict['kappa_m'][i][0])
                self.partial_kappa_m[i].append(metrics_dict['kappa_m'][i][1])
                self.line_global_kappa_m[i].set_data(self.X, self.global_kappa_m[i])
                self.line_partial_kappa_m[i].set_data(self.X, self.partial_kappa_m[i])

                self.temp.append(self.subplot_kappa_m.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_kappa_m[i][-1],
                                                                self.partial_kappa_m[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y
                minimum = min(min(minimum, min(self.global_kappa_m[i])), min(minimum, min(self.partial_kappa_m[i])))

            self.subplot_kappa_m.set_xlim(0, self.X[-1])
            self.subplot_kappa_m.set_ylim(minimum, 1.)

        if 'hamming_score' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_hamming_score.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_hamming_score[i].append(metrics_dict['hamming_score'][i][0])
                self.partial_hamming_score[i].append(metrics_dict['hamming_score'][i][1])
                self.line_global_hamming_score[i].set_data(self.X, self.global_hamming_score[i])
                self.line_partial_hamming_score[i].set_data(self.X, self.partial_hamming_score[i])

                self.temp.append(self.subplot_hamming_score.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_hamming_score[i][-1],
                                                                self.partial_hamming_score[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y

            self.subplot_hamming_score.set_xlim(0, self.X[-1])
            self.subplot_hamming_score.set_ylim(0, 1)

        if 'hamming_loss' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_hamming_loss.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_hamming_loss[i].append(metrics_dict['hamming_loss'][i][0])
                self.partial_hamming_loss[i].append(metrics_dict['hamming_loss'][i][1])
                self.line_global_hamming_loss[i].set_data(self.X, self.global_hamming_loss[i])
                self.line_partial_hamming_loss[i].set_data(self.X, self.partial_hamming_loss[i])

                self.temp.append(self.subplot_hamming_loss.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_hamming_loss[i][-1],
                                                                self.partial_hamming_loss[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y

            self.subplot_hamming_loss.set_xlim(0, self.X[-1])
            self.subplot_hamming_loss.set_ylim(0, 1)

        if 'exact_match' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_exact_match.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_exact_match[i].append(metrics_dict['exact_match'][i][0])
                self.partial_exact_match[i].append(metrics_dict['exact_match'][i][1])
                self.line_global_exact_match[i].set_data(self.X, self.global_exact_match[i])
                self.line_partial_exact_match[i].set_data(self.X, self.partial_exact_match[i])

                self.temp.append(self.subplot_exact_match.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_exact_match[i][-1],
                                                                self.partial_exact_match[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y

            self.subplot_exact_match.set_xlim(0, self.X[-1])
            self.subplot_exact_match.set_ylim(0, 1)

        if 'j_index' in self.plots:
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_j_index.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_j_index[i].append(metrics_dict['j_index'][i][0])
                self.partial_j_index[i].append(metrics_dict['j_index'][i][1])
                self.line_global_j_index[i].set_data(self.X, self.global_j_index[i])
                self.line_partial_j_index[i].set_data(self.X, self.partial_j_index[i])

                self.temp.append(self.subplot_j_index.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_j_index[i][-1],
                                                                self.partial_j_index[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y

            self.subplot_j_index.set_xlim(0, self.X[-1])
            self.subplot_j_index.set_ylim(0, 1)

        if 'mean_square_error' in self.plots:
            minimum = -1
            maximum = 0
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_mse.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_mse[i].append(metrics_dict['mean_square_error'][i][0])
                self.partial_mse[i].append(metrics_dict['mean_square_error'][i][1])
                self.line_global_mse[i].set_data(self.X, self.global_mse[i])
                self.line_partial_mse[i].set_data(self.X, self.partial_mse[i])

                self.temp.append(self.subplot_mse.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_mse[i][-1],
                                                                self.partial_mse[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y
                # minimum = min([min(self.global_mse[i]), min(self.partial_mse[i]), minimum])
                maximum = max([max(self.global_mse[i]), max(self.partial_mse[i]), maximum])

            self.subplot_mse.set_xlim(0, self.X[-1])
            self.subplot_mse.set_ylim(minimum, 1.2*maximum)

        if 'mean_absolute_error' in self.plots:
            minimum = -1
            maximum = 0
            shift_y = 10  # y axis shift for plot annotations
            xy_pos = xy_pos_default
            self.temp.append(self.subplot_mae.annotate(
                '{: <8} | {: ^14} | {: ^14}'.format('Learner', 'Global', 'Sliding'),
                xy=xy_pos, xycoords='axes fraction'))
            for i in range(self.n_learners):
                self.global_mae[i].append(metrics_dict['mean_absolute_error'][i][0])
                self.partial_mae[i].append(metrics_dict['mean_absolute_error'][i][1])
                self.line_global_mae[i].set_data(self.X, self.global_mae[i])
                self.line_partial_mae[i].set_data(self.X, self.partial_mae[i])

                self.temp.append(self.subplot_mae.annotate(
                    '{: ^12d}   {: ^14.3f}   {: ^14.3f}'.format(i,
                                                                self.global_mae[i][-1],
                                                                self.partial_mae[i][-1]),
                    xy=xy_pos, xycoords='axes fraction',
                    xytext=(0, -shift_y),
                    textcoords='offset points'))
                shift_y += shift_y
                # minimum = min([min(self.global_mae[i]), min(self.partial_mae[i]), minimum])
                maximum = max([max(self.global_mae[i]), max(self.partial_mae[i]), maximum])

            self.subplot_mae.set_xlim(0, self.X[-1])
            self.subplot_mae.set_ylim(minimum, 1.2*maximum)

        if 'true_vs_predicts' in self.plots:
            self.true_values.append(metrics_dict['true_vs_predicts'][0][0])
            self.line_true.set_data(self.X, self.true_values)
            minimum = 0
            maximum = 0
            for i in range(self.n_learners):
                self.pred_values[i].append(metrics_dict['true_vs_predicts'][i][1])
                self.line_pred[i].set_data(self.X, self.pred_values[i])
                minimum = min([min(self.pred_values[i]), min(self.true_values), minimum])
                maximum = max([max(self.pred_values[i]), max(self.true_values), maximum])

            self.subplot_true_vs_predicts.set_xlim(0, self.X[-1])
            self.subplot_true_vs_predicts.set_ylim(minimum - 1, maximum + 1)

            self.subplot_true_vs_predicts.legend(loc=2, bbox_to_anchor=(1.01, 1.))

        if self._draw_cnt == 4:
            plt.subplots_adjust(right=0.72)   # Adjust subplots to include metrics
            self.fig.canvas.draw()
            plt.pause(1e-9)
            self._draw_cnt = 0
        else:
            self._draw_cnt += 1

    def draw_scatter_points(self, X, y, predict):
        pass

    @staticmethod
    def hold():
        plt.show(block=True)

    def get_info(self):
        pass


if __name__ == "__main__":
    ev = EvaluationVisualizer()
    print(ev.get_class_type())
