import warnings
import time
from skmultiflow.visualization.base_listener import BaseListener
from matplotlib.rcsetup import cycler
import matplotlib.pyplot as plt
from matplotlib import get_backend
from skmultiflow.utils import FastBuffer, constants


class EvaluationVisualizer(BaseListener):
    """ This class is responsible for maintaining and updating the plot modules
    for the evaluators in scikit-multiflow.
    
    It uses `matplotlib.pyplot` modules to create the main plot, which
    depending on the options passed to it as parameter, will create multiple 
    subplots to better display all requested metrics.
    
    The plots are dynamically updated depending on frame counts amd time elapsed
    since last update. This strategy account for fast and slow methods.
    
    Line objects are used to describe performance measurements.
    
    It supports multiple models per subplot as a way of comparing the performance
    of different learning algorithms.
    
    Parameters
    ----------
    n_wait: int
        The number of samples tracked in the sliding window for current performance.
    
    dataset_name: string (Default: 'Unnamed graph')
        The title of the plot. Algorithmically it's not important.
    
    metrics: list
        A list containing all the metrics to plot.
    
    n_models: int
        The number of models to compare.
    
    Raises
    ------
    ValueError: A ValueError can be raised for a series of reasons. If no plots 
    are passed as parameter to the constructor a ValueError is raised. If the wrong 
    type of parameter is passed to on_new_train_step the same error is raised.
    
    Notes
    -----
    Using more than 3 plot types at a time is not recommended, as it can
    significantly slow down the evaluation time. For the same reason comparing
    more than 3 learners at a time is not recommended.
    
    """

    def __init__(self, task_type, n_wait, dataset_name, metrics, n_models, model_names, data_dict):
        super().__init__()

        # Default values
        self._sample_ids = []
        self._is_legend_set = False
        self._frame_cnt = 0
        self._plot_trackers = {}
        self._text_annotations = []
        self._last_draw_timestamp = 0

        # Configuration
        self.data_dict = data_dict
        self.n_wait = n_wait
        self.dataset_name = dataset_name
        self.n_models = n_models

        # Validate inputs
        if task_type is None or task_type == constants.UNDEFINED:
            raise ValueError('Task type for visualizer object is undefined.')
        else:
            if task_type in [constants.CLASSIFICATION, constants.REGRESSION, constants.MULTI_TARGET_CLASSIFICATION,
                             constants.MULTI_TARGET_REGRESSION]:
                self.task_type = task_type
            else:
                raise ValueError('Invalid task type: {}'.format(task_type))

        if model_names is None:
            self.model_names = ['M{}'.format(i) for i in range(n_models)]
        else:
            if isinstance(model_names, list):
                if len(model_names) != n_models:
                    raise ValueError("Number of model names {} does not match the number of models {}.".
                                     format(len(model_names), n_models))
                else:
                    self.model_names = model_names
            else:
                raise ValueError("model_names must be a list.")

        if metrics is not None:
            if len(metrics) < 1:
                raise ValueError('The metrics list is empty.')
            else:
                self.metrics = metrics
        else:
            raise ValueError('Invalid metrics {}'.format(metrics))

        if constants.DATA_POINTS in metrics and n_models > 1:
            raise ValueError("Can not use multiple models with data points visualization.")

        # Proceed with configuration
        self.__configure()

    def on_new_train_step(self, sample_id, data_buffer):
        """ This is the listener main function, which gives it the ability to
        'listen' for the caller. Whenever the EvaluationVisualiser should 
        be aware of some new data, the caller will invoke this function,
        passing the new data buffer.
        
        Parameters
        ----------
        sample_id: int
            The current sample id.

        data_buffer: EvaluationDataBuffer
            A buffer containing evaluation data for a single training / visualization step.
            
        Raises
        ------
        ValueError: If an exception is raised during the draw operation.
         
        """

        try:
            current_time = time.time()
            self._clear_annotations()
            self._update_plots(sample_id, data_buffer)

            # To mitigate re-drawing overhead for fast models use frame counter (default = 5 frames).
            # To avoid slow refresh rate in slow models use a time limit (default = 1 sec).
            if (self._frame_cnt == 5) or (current_time - self._last_draw_timestamp > 1):
                plt.subplots_adjust(right=0.72, bottom=0.22)  # Adjust subplots to include metrics annotations
                if get_backend() == 'nbAgg':
                    self.fig.canvas.draw()    # Force draw in'notebook' backend
                plt.pause(1e-9)
                self._frame_cnt = 0
                self._last_draw_timestamp = current_time
            else:
                self._frame_cnt += 1
        except BaseException as exception:
            raise ValueError('Failed when trying to draw plot. Exception: {} | Type: {}'.
                             format(exception, type(exception).__name__))

    def __configure(self):
        """  This function will verify which subplots should be create. Initializing
        all relevant objects to keep track of the plotting points.
        
        Basic structures needed to keep track of plot values (for each subplot) 
        are: lists of values and matplot line objects.
        
        The __configure function will also initialize each subplot with the 
        correct name and setup the axis.
        
        The subplot size will self adjust to each screen size, so that data can 
        be better viewed in different contexts.

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

        self._sample_ids = []
        memory_time = {}

        plt.ion()
        self.fig = plt.figure(figsize=(9, 5))
        self.fig.suptitle(self.dataset_name)
        plot_metrics = [m for m in self.metrics if m not in [constants.RUNNING_TIME, constants.MODEL_SIZE]]
        base = 11 + len(plot_metrics) * 100  # 3-digit integer describing the position of the subplot.
        self.fig.canvas.set_window_title('scikit-multiflow')

        # Subplots handler
        for metric_id in self.metrics:
            data_ids = self.data_dict[metric_id]
            self._plot_trackers[metric_id] = PlotDataTracker(data_ids)
            plot_tracker = self._plot_trackers[metric_id]
            if metric_id not in [constants.RUNNING_TIME, constants.MODEL_SIZE]:
                plot_tracker.sub_plot_obj = self.fig.add_subplot(base)
            base += 1
            if metric_id == constants.TRUE_VS_PREDICTED:
                handle = []
                plot_tracker.sub_plot_obj.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))
                for data_id in data_ids:
                    if data_id == constants.Y_TRUE:
                        # True data
                        plot_tracker.data[data_id] = []
                        label = 'True value'
                        line_style = '--'
                        line_obj = plot_tracker.line_objs
                        if self.task_type == constants.CLASSIFICATION:
                            line_obj[data_id], = plot_tracker.sub_plot_obj.step(self._sample_ids,
                                                                                plot_tracker.data[data_id],
                                                                                label=label, linestyle=line_style)
                        else:
                            line_obj[data_id], = plot_tracker.sub_plot_obj.plot(self._sample_ids,
                                                                                plot_tracker.data[data_id],
                                                                                label=label, linestyle=line_style)
                        handle.append(line_obj[data_id])
                    else:
                        # Predicted data
                        plot_tracker.data[data_id] = [[] for _ in range(self.n_models)]
                        plot_tracker.line_objs[data_id] = [None for _ in range(self.n_models)]
                        line_obj = plot_tracker.line_objs[data_id]
                        for i in range(self.n_models):
                            label = 'Predicted {}'.format(self.model_names[i])
                            line_style = '--'
                            if self.task_type == constants.CLASSIFICATION:
                                line_obj[i], = plot_tracker.sub_plot_obj.step(self._sample_ids,
                                                                              plot_tracker.data[data_id][i],
                                                                              label=label, linestyle=line_style)
                            else:
                                line_obj[i], = plot_tracker.sub_plot_obj.plot(self._sample_ids,
                                                                              plot_tracker.data[data_id][i],
                                                                              label=label, linestyle=line_style)
                            handle.append(line_obj[i])
                plot_tracker.sub_plot_obj.legend(handles=handle, loc=2, bbox_to_anchor=(1.01, 1.))
                plot_tracker.sub_plot_obj.set_title('True vs Predicted')
                plot_tracker.sub_plot_obj.set_ylabel('y')

            elif metric_id == constants.DATA_POINTS:
                plot_tracker.data['buffer_size'] = 100
                plot_tracker.data['X'] = FastBuffer(plot_tracker.data['buffer_size'])
                plot_tracker.data['target_values'] = None
                plot_tracker.data['predictions'] = FastBuffer(plot_tracker.data['buffer_size'])
                plot_tracker.data['clusters'] = []
                plot_tracker.data['clusters_initialized'] = False

            elif metric_id == constants.RUNNING_TIME:
                # Only the current time measurement must be saved
                for data_id in data_ids:
                    plot_tracker.data[data_id] = [0.0 for _ in range(self.n_models)]
                # To make the annotations
                memory_time.update(plot_tracker.data)

            elif metric_id == constants.MODEL_SIZE:
                plot_tracker.data['model_size'] = [0.0 for _ in range(self.n_models)]

                memory_time['model_size'] = plot_tracker.data['model_size']
            else:
                # Default case, 'mean' and 'current' performance
                handle = []
                sorted_data_ids = data_ids.copy()
                sorted_data_ids.sort()    # For better usage of the color cycle, start with 'current' data
                for data_id in sorted_data_ids:
                    plot_tracker.data[data_id] = [[] for _ in range(self.n_models)]
                    plot_tracker.line_objs[data_id] = [None for _ in range(self.n_models)]
                    line_obj = plot_tracker.line_objs[data_id]
                    for i in range(self.n_models):
                        if data_id == constants.CURRENT:
                            label = '{}  (current, {} samples)'.format(self.model_names[i], self.n_wait)
                            line_style = '-'
                        else:
                            label = '{} (mean)'.format(self.model_names[i])
                            line_style = ':'
                        line_obj[i], = plot_tracker.sub_plot_obj.plot(self._sample_ids,
                                                                      plot_tracker.data[data_id][i],
                                                                      label=label,
                                                                      linestyle=line_style)
                        handle.append(line_obj[i])
                self._set_fig_legend(handle)

                if metric_id == constants.ACCURACY:
                    plot_tracker.sub_plot_obj.set_title('Accuracy')
                    plot_tracker.sub_plot_obj.set_ylabel('acc')
                elif metric_id == constants.KAPPA:
                    plot_tracker.sub_plot_obj.set_title('Kappa')
                    plot_tracker.sub_plot_obj.set_ylabel('kappa')
                elif metric_id == constants.KAPPA_T:
                    plot_tracker.sub_plot_obj.set_title('Kappa T')
                    plot_tracker.sub_plot_obj.set_ylabel('kappa t')
                elif metric_id == constants.KAPPA_M:
                    plot_tracker.sub_plot_obj.set_title('Kappa M')
                    plot_tracker.sub_plot_obj.set_ylabel('kappa m')
                elif metric_id == constants.PRECISION:
                    plot_tracker.sub_plot_obj.set_title('Precision')
                    plot_tracker.sub_plot_obj.set_ylabel('precision')
                elif metric_id == constants.RECALL:
                    plot_tracker.sub_plot_obj.set_title('Recall')
                    plot_tracker.sub_plot_obj.set_ylabel('recall')
                elif metric_id == constants.F1_SCORE:
                    plot_tracker.sub_plot_obj.set_title('F1 Score')
                    plot_tracker.sub_plot_obj.set_ylabel('f1 score')
                elif metric_id == constants.HAMMING_SCORE:
                    plot_tracker.sub_plot_obj.set_title('Hamming score')
                    plot_tracker.sub_plot_obj.set_ylabel('hamming score')
                elif metric_id == constants.HAMMING_LOSS:
                    plot_tracker.sub_plot_obj.set_title('Hamming loss')
                    plot_tracker.sub_plot_obj.set_ylabel('hamming loss')
                elif metric_id == constants.EXACT_MATCH:
                    plot_tracker.sub_plot_obj.set_title('Exact Match')
                    plot_tracker.sub_plot_obj.set_ylabel('exact match')
                elif metric_id == constants.J_INDEX:
                    plot_tracker.sub_plot_obj.set_title('Jaccard Index')
                    plot_tracker.sub_plot_obj.set_ylabel('j-index')
                elif metric_id == constants.MSE:
                    plot_tracker.sub_plot_obj.set_title('Mean Squared Error')
                    plot_tracker.sub_plot_obj.set_ylabel('mse')
                elif metric_id == constants.MAE:
                    plot_tracker.sub_plot_obj.set_title('Mean Absolute Error')
                    plot_tracker.sub_plot_obj.set_ylabel('mae')
                elif metric_id == constants.AMSE:
                    plot_tracker.sub_plot_obj.set_title('Average Mean Squared Error')
                    plot_tracker.sub_plot_obj.set_ylabel('amse')
                elif metric_id == constants.AMAE:
                    plot_tracker.sub_plot_obj.set_title('Average Mean Absolute Error')
                    plot_tracker.sub_plot_obj.set_ylabel('amae')
                elif metric_id == constants.ARMSE:
                    plot_tracker.sub_plot_obj.set_title('Average Root Mean Squared Error')
                    plot_tracker.sub_plot_obj.set_ylabel('armse')
                elif metric_id == constants.DATA_POINTS:
                    plot_tracker.sub_plot_obj.set_title('')
                    plot_tracker.sub_plot_obj.set_xlabel('Feature x')
                    plot_tracker.sub_plot_obj.set_ylabel('Feature y')
                else:
                    plot_tracker.sub_plot_obj.set_title('Unknown metric')
                    plot_tracker.sub_plot_obj.set_ylabel('')

        if constants.DATA_POINTS not in self.metrics:
            plt.xlabel('Samples')
        if constants.RUNNING_TIME in self.metrics or \
                constants.MODEL_SIZE in self.metrics:
            self._update_time_and_memory_annotations(memory_time)

        self.fig.subplots_adjust(hspace=.5)
        self.fig.tight_layout(rect=[0, .04, 1, 0.98], pad=2.6, w_pad=0.4, h_pad=1.0)

    def _set_fig_legend(self, handles=None):
        if not self._is_legend_set:
            self.fig.legend(handles=handles, ncol=2, bbox_to_anchor=(0.98, 0.04), loc="lower right")
            self._is_legend_set = True

    def _update_plots(self, sample_id, data_buffer):
        self._sample_ids.append(sample_id)
        memory_time = {}
        for metric_id, data_ids in data_buffer.data_dict.items():
            # update_xy_limits = True
            update_xy_limits = metric_id not in [constants.RUNNING_TIME, constants.MODEL_SIZE]
            y_min = 0.0
            y_max = 1.0
            pad = 0.1  # Default padding to set above and bellow plots
            plot_tracker = self._plot_trackers[metric_id]
            if metric_id == constants.TRUE_VS_PREDICTED:
                # Process true values
                data_id = constants.Y_TRUE
                plot_tracker.data[data_id].append(data_buffer.get_data(metric_id=metric_id, data_id=data_id))
                plot_tracker.line_objs[data_id].set_data(self._sample_ids, plot_tracker.data[data_id])
                # Process predicted values
                data_id = constants.Y_PRED
                data = data_buffer.get_data(metric_id=metric_id, data_id=data_id)
                for i in range(self.n_models):
                    plot_tracker.data[data_id][i].append(data[i])
                    plot_tracker.line_objs[data_id][i].set_data(self._sample_ids, plot_tracker.data[data_id][i])
                    y_min = min([plot_tracker.data[data_id][i][-1], plot_tracker.data[constants.Y_TRUE][-1], y_min])
                    y_max = max([plot_tracker.data[data_id][i][-1], plot_tracker.data[constants.Y_TRUE][-1], y_max])
            elif metric_id == constants.DATA_POINTS:
                update_xy_limits = False
                # Process features
                data_id = 'X'
                features_dict = data_buffer.get_data(metric_id=metric_id, data_id=data_id)
                feature_indices = list(features_dict.keys())
                feature_indices.sort()
                # Store tuple of feature values into the buffer, sorted by index
                plot_tracker.data[data_id].add_element([[features_dict[feature_indices[0]],
                                                        features_dict[feature_indices[1]]]])

                plot_tracker.sub_plot_obj.set_xlabel('Feature {}'.format(feature_indices[0]))
                plot_tracker.sub_plot_obj.set_ylabel('Feature {}'.format(feature_indices[1]))
                # TODO consider a fading/update strategy instead
                plot_tracker.sub_plot_obj.clear()
                X1 = plot_tracker.data[data_id].get_queue()[-1][0]
                X2 = plot_tracker.data[data_id].get_queue()[-1][1]

                # Process target values
                data_id = 'target_values'
                plot_tracker.data[data_id] = data_buffer.get_data(metric_id=metric_id, data_id=data_id)
                if not plot_tracker.data['clusters_initialized']:
                    for j in range(len(plot_tracker.data[data_id])):
                        plot_tracker.data['clusters'].append(FastBuffer(plot_tracker.data['buffer_size']))

                # Process predictions
                data_id = 'predictions'
                plot_tracker.data[data_id].add_element([data_buffer.get_data(metric_id=metric_id, data_id=data_id)])

                for k, cluster in enumerate(plot_tracker.data['clusters']):
                    if plot_tracker.data[data_id].get_queue()[-1] == k:
                        plot_tracker.data['clusters'][k].add_element([(X1, X2)])
                        # TODO confirm buffer update inside the loop
                    if cluster.get_queue():
                        temp = cluster.get_queue()
                        plot_tracker.sub_plot_obj.scatter(*zip(*temp), label="Class {k}".format(k=k))
                plot_tracker.sub_plot_obj.legend(loc=2, bbox_to_anchor=(1.01, 1.))
            elif metric_id == constants.RUNNING_TIME:
                # Only the current time measurement must be saved
                for data_id in data_ids:
                    plot_tracker.data[data_id] = data_buffer.get_data(
                        metric_id=metric_id,
                        data_id=data_id
                    )
                memory_time.update(plot_tracker.data)

            elif metric_id == constants.MODEL_SIZE:
                plot_tracker.data['model_size'] = data_buffer.get_data(
                    metric_id=metric_id,
                    data_id='model_size'
                )
                memory_time['model_size'] = plot_tracker.data['model_size']
            else:
                # Default case, 'mean' and 'current' performance
                for data_id in data_ids:
                    # Buffer data
                    data = data_buffer.get_data(metric_id=metric_id, data_id=data_id)
                    for i in range(self.n_models):
                        plot_tracker.data[data_id][i].append(data[i])
                        plot_tracker.line_objs[data_id][i].set_data(self._sample_ids, plot_tracker.data[data_id][i])
                # Process data
                for i in range(self.n_models):
                    # Update annotations
                    self._update_annotations(i, plot_tracker.sub_plot_obj, self.model_names[i],
                                             plot_tracker.data[constants.MEAN][i][-1],
                                             plot_tracker.data[constants.CURRENT][i][-1])
                    # Update plot limits
                    if metric_id in [constants.KAPPA_T, constants.KAPPA_M]:
                        y_min = min([plot_tracker.data[constants.MEAN][i][-1],
                                     plot_tracker.data[constants.CURRENT][i][-1], y_min])
                    if metric_id in [constants.MSE, constants.MAE, constants.AMSE, constants.AMAE, constants.ARMSE]:
                        y_min = -1
                        y_max = max([plot_tracker.data[constants.MEAN][i][-1],
                                     plot_tracker.data[constants.CURRENT][i][-1], y_max])
                        pad = 0.5 * y_max  # Padding bellow and above thresholds
            if update_xy_limits:
                plot_tracker.sub_plot_obj.set_ylim((y_min-pad, y_max+pad))
                plot_tracker.sub_plot_obj.set_xlim(0, self._sample_ids[-1])
        if constants.RUNNING_TIME in self.metrics or \
                constants.MODEL_SIZE in self.metrics:
            self._update_time_and_memory_annotations(memory_time)

    def _clear_annotations(self):
        """ Clear annotations, so next frame is correctly rendered. """
        for i in range(len(self._text_annotations)):
            self._text_annotations[i].remove()
        self._text_annotations = []

    def _update_annotations(self, idx, subplot, model_name, mean_value, current_value):
        xy_pos_default = (1.02, .90)  # Default xy position for metric annotations
        shift_y = 10 * (idx + 1)  # y axis shift for plot annotations
        xy_pos = xy_pos_default
        if idx == 0:
            self._text_annotations.append(subplot.annotate('{: <12} | {: ^16} | {: ^16}'.
                                                           format('Model', 'Mean', 'Current'),
                                                           xy=xy_pos, xycoords='axes fraction'))
        self._text_annotations.append(subplot.annotate('{: <10.10s}'.format(model_name[:6]),
                                                       xy=xy_pos, xycoords='axes fraction',
                                                       xytext=(0, -shift_y), textcoords='offset points'))
        self._text_annotations.append(subplot.annotate('{: ^16.4f}  {: ^16.4f}'.format(mean_value, current_value),
                                                       xy=xy_pos, xycoords='axes fraction',
                                                       xytext=(50, -shift_y), textcoords='offset points'))

    def _update_time_and_memory_annotations(self, memory_time):
        text_header = '{: <12s}'.format('Model')
        if constants.RUNNING_TIME in self.metrics:
            text_header += ' | {: ^16s} | {: ^16s} | {: ^16s}'.\
                      format('Train (s)', 'Predict (s)', 'Total (s)')
        if constants.MODEL_SIZE in self.metrics:
            text_header += ' | {: ^16}'.format('Mem (kB)')

        last_plot = self.fig.get_axes()[-1]
        x0, y0, width, height = last_plot.get_position().bounds
        annotation_xy = (.1, .1)
        self._text_annotations.append(self.fig.text(s=text_header, x=annotation_xy[0], y=annotation_xy[1]))

        for i, m_name in enumerate(self.model_names):
            text_info = '{: <15s}'.format(m_name[:6])
            if constants.RUNNING_TIME in self.metrics:
                text_info += '{: ^19.2f}  {: ^19.2f}  {: ^19.2f}  '.format(memory_time['training_time'][i],
                                                                           memory_time['testing_time'][i],
                                                                           memory_time['total_running_time'][i])
            if constants.MODEL_SIZE in self.metrics:
                text_info += '{: ^19.2f}'.format(memory_time['model_size'][i])
            shift_y = .03 * (i + 1)  # y axis shift for plot annotations
            self._text_annotations.append(self.fig.text(s=text_info, x=annotation_xy[0], y=annotation_xy[1] - shift_y))

    @staticmethod
    def hold():
        plt.show(block=True)


class PlotDataTracker(object):
    """ A class to track relevant data for plots corresponding to selected metrics.
    Data buffers and line objects are accessible via the corresponding data_id.

    """
    def __init__(self, data_ids: list):
        self.data_ids = None
        self.data = {}
        self.line_objs = {}
        self.sub_plot_obj = None
        self._validate(data_ids)
        self._configure()

    def _validate(self, data_ids):
        if isinstance(data_ids, list):
            if len(data_ids) > 0:
                self.data_ids = data_ids
            else:
                raise ValueError('data_ids is empty')
        else:
            raise TypeError('data_ids must be a list, received: {}'.format(type(data_ids)))

    def _configure(self):
        for data_id in self.data_ids:
            self.data[data_id] = None
            self.line_objs[data_id] = None
