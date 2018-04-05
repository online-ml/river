__author__ = 'Guilherme Matsumoto'

from skmultiflow.data import base_instance_stream
from skmultiflow.core.base_object import BaseObject
import pandas as pd
import numpy as np


class FileStream(base_instance_stream.BaseInstanceStream, BaseObject):
    """ FileStream
    
    A stream generated from the entries of a file. For the moment only 
    csv files are supported, but the idea is to support any file format, 
    as long as there is a function that correctly reads, interprets, and 
    returns a pandas' DataFrame or numpy.ndarray with the data.
    
    The stream is able to provide, as requested, a number of samples, in 
    a way that old samples cannot be accessed in a later time. This is done 
    so that a stream context can be correctly simulated. 
    
    Parameters
    ----------
    file_opt: FileOption object
        Holds the options relative to the file to be read. For a detailed 
        documentation please refer to: skmultiflow.options.file_option.
        
    targets_index: int
        The index from which the targets (labels) start.
        
    num_target_tasks: int
        The number of targeting tasks.
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.options.file_option import FileOption
    >>> from skmultiflow.data.file_stream import FileStream
    >>> # Setup the stream
    >>> file_option = FileOption('FILE', 'sea', 'skmultiflow/datasets/sea_stream.csv', 'csv', False)
    >>> stream = FileStream(file_option)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_instance()
    (array([[0.080429, 8.397187, 7.074928]]), array([0]))
    >>> # Retrieving 10 samples
    >>> stream.next_instance(10)
    (array([[1.42074 , 7.504724, 6.764101],
        [0.960543, 5.168416, 8.298959],
        [3.367279, 6.797711, 4.857875],
        [9.265933, 8.548432, 2.460325],
        [7.295862, 2.373183, 3.427656],
        [9.289001, 3.280215, 3.154171],
        [0.279599, 7.340643, 3.729721],
        [4.387696, 1.97443 , 6.447183],
        [2.933823, 7.150514, 2.566901],
        [4.303049, 1.471813, 9.078151]]),
        array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0]))
    >>> stream.estimated_remaining_instances()
    39989
    >>> stream.has_more_instances()
    True

    """

    def __init__(self, file_opt, targets_index=-1, num_target_tasks=1):
        super().__init__()
        # default values
        if str(file_opt.file_type).lower() == 'csv':
            self.read_function = pd.read_csv
        else:
            raise ValueError('Unsupported format: ', file_opt.file_type)
        self.file_name = None
        self.instance_index = 0
        self.instance_length = 0
        self.X = None
        self.y = None
        self.current_instance_x = None
        self.current_instance_y = None
        self.num_target_tasks = 1
        self.target_index = -1
        self.num_numerical_attributes = 0
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.attributes_header = None
        self.classes_header = None
        self.instances = None
        self.num_attributes = 0
        self.num_classes = 0

        self.__configure(file_opt, targets_index, num_target_tasks)

    def __configure(self, file_opt, targets_index, num_target_tasks):
        self.file_name = file_opt.get_file_name()
        self.target_index = targets_index
        self.instances = None
        self.instance_length = 0
        self.current_instance_x = None
        self.current_instance_y = None
        self.num_target_tasks = num_target_tasks
        self.X = None
        self.y = None

    def prepare_for_use(self):
        """ prepare_for_use
        
        Prepares the stream for use. This functions should always be 
        called after the stream initialization.
        
        """
        self._load_data()
        self.restart()

    def _load_data(self):
        try:
            instance_aux = self.read_function(self.file_name)
            self.instance_length = len(instance_aux.index)
            self.num_attributes = len(instance_aux.columns) - 1
            self.num_numerical_attributes = self.num_attributes
            labels = instance_aux.columns.values.tolist()

            if (self.target_index + self.num_target_tasks == len(labels)) \
                    or (self.target_index + self.num_target_tasks == 0):
                self.y = instance_aux.iloc[:, self.target_index:].as_matrix()
                y_labels = labels[self.target_index:]
                self.classes_header = labels[self.target_index:]
                self.attributes_header = labels[:self.target_index]
            else:
                self.y = instance_aux.iloc[:, self.target_index:self.target_index+self.num_target_tasks].as_matrix()
                y_labels = labels[self.target_index:self.target_index+self.num_target_tasks]
                self.classes_header = labels[self.num_target_tasks:self.target_index+self.num_target_tasks]
                self.attributes_header = labels[:self.target_index]
                self.attributes_header.extend(labels[self.target_index + self.num_target_tasks:])

            self.X = instance_aux.drop(y_labels, axis=1).as_matrix()

            self.num_classes = len(np.unique(self.y))

        except IOError:
            print("{} file reading failed.".format(self.file_name))
        pass

    def restart(self):
        """ restart
        
        Restarts the stream's sample feeding, while keeping all of its 
        parameters.
        
        It basically server the purpose of reinitializing the stream to 
        its initial state.
        
        """
        self.instance_index = 0
        self.current_instance_x = None
        self.current_instance_y = None

    def is_restartable(self):
        return True

    def next_instance(self, batch_size=1):
        """ next_instance
        
        If there is enough instances to supply at least batch_size samples, those 
        are returned. If there aren't a tuple of (None, None) is returned.
        
        Parameters
        ----------
        batch_size: int
            The number of instances to return.
        
        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.
        
        """
        self.instance_index += batch_size
        try:

            self.current_instance_x = self.X[self.instance_index - batch_size:self.instance_index, :]
            self.current_instance_y = self.y[self.instance_index - batch_size:self.instance_index, :]
            if self.num_target_tasks < 2:
                self.current_instance_y = self.current_instance_y.flatten()

        except IndexError:
            self.current_instance_x = None
            self.current_instance_y = None
        return self.current_instance_x, self.current_instance_y

    def has_more_instances(self):
        return (self.instance_length - self.instance_index) > 0

    def estimated_remaining_instances(self):
        return self.instance_length - self.instance_index

    def print_df(self):
        print(self.X)
        print(self.y)

    def get_instances_length(self):
        return self.instance_length

    def get_num_attributes(self):
        return self.num_attributes

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_targets(self):
        return self.num_target_tasks

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.classes_header

    def get_last_instance(self):
        return self.current_instance_x, self.current_instance_y

    def get_plot_name(self):
        aux = self.file_name.split("/")
        if aux[len(aux)-1] == '':
            aux.pop(len(aux)-1)
        return "{} - {} class labels".format(aux[len(aux)-1], self.num_classes) \
            if self.num_target_tasks == 1 \
            else "{} - {} classification tasks".format(aux[len(aux)-1], self.num_target_tasks)

    def get_classes(self):
        c = np.unique(self.y).tolist()
        return c

    def get_info(self):
        return 'File Stream: file_name: ' + str(self.file_name) + \
               '  -  num_classes: ' + str(self.num_classes) + \
               '  -  num_classification_tasks: ' + str(self.num_target_tasks)

    def get_num_targeting_tasks(self):
        return self.num_target_tasks
