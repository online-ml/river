__author__ = 'Jacob Montiel'

from skmultiflow.data import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
from skmultiflow.core.BaseObject import BaseObject
import pandas as pd
import numpy as np

# Deprecated, use FileStream
class FileToStream(BaseInstanceStream.BaseInstanceStream, BaseObject):
    '''
        Converts a data file into a data stream
        -------------------------------------------
        Generates a stream from a file
        
        Parser parameters
        ---------------------------------------------
        -f: file
    '''
    def __init__(self, file_opt, num_classes = 2):
        super().__init__()
        '''
        __init__(self, file_name, index)
        
        Parameters
        ----------------------------------------
        file_name : string
                   Name of the file
        index : int
                Class index parameter
        '''
        self.file_name = file_opt.get_file_name()
        if file_opt.file_type is "CSV":
            self.read_function = pd.read_csv
        else:
            raise ValueError('Unsupported format: ', file_opt.file_type)
        self.instance_index = 0
        self.instances = None
        self.num_instances = 0
        self.current_instance = None
        self.num_attributes = 0
        self.num_classes = num_classes
        self.num_numerical_attributes = 0
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.attributes_header = None
        self.classes_header = None

    def prepare_for_use(self):
        self.restart()

    def restart(self):
        '''
        restart(self)
        ----------------------------------------
        Read the file and set object attributes
        '''
        if not self.instances:
            try:
                self.instances = self.read_function(self.file_name)
                self.num_instances = self.instances.shape[0]
                self.num_attributes = self.instances.shape[1] - self.num_classes
                labels = self.instances.columns.values.tolist()
                self.attributes_header = labels[0:(len(labels) - self.num_classes)]
                self.classes_header = labels[(len(labels) - self.num_classes):]
            except IOError:
                print("File reading failed.")
        else:
            self.instance_index = 0
        pass

    def is_restartable(self):
        return True

    def next_instance(self, batch_size = 1):
        self.current_instance = Instance(self.num_attributes,
                                         self.num_classes, -1,
                                        self.instances.iloc[self.instance_index:self.instance_index + 1].values[0])
        self.instance_index += 1
        return self.current_instance

    def has_more_instances(self):
        return ((self.num_instances - self.instance_index) > 0)

    def estimated_remaining_instances(self):
        return (self.num_instances - self.instance_index)

    def print_df(self):
        print(self.instances)

    def get_instances_length(self):
        return self.num_instances

    def get_num_attributes(self):
        return self.num_attributes

    def has_more_mini_batch(self):
        pass

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_classes(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.classes_header

    def get_plot_name(self):
        return "File Stream - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = np.unique(self.instances[:, self.num_attributes:])
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def get_info(self):
        pass