__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from sklearn.datasets import make_multilabel_classification
import numpy as np


class MultilabelGenerator(BaseInstanceStream):
    def __init__(self, n_samples=40000, n_features=20, n_targets=5, n_labels=2):
        super().__init__()
        self.X = None
        self.y = None
        self.num_samples = 0
        self.num_features = 0
        self.num_target_tasks = 0
        self.num_labels = 0
        self.instance_index = 0
        self.current_instance_y = None
        self.current_instance_x = None
        self.configure(n_samples, n_features, n_targets, n_labels)
        pass

    def configure(self, n_samples, n_features, n_targets, n_labels):
        self.X, self.y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_targets, n_labels=n_labels)
        self.num_samples = n_samples
        self.num_features = n_features
        self.num_target_tasks = n_targets
        self.num_labels = n_labels
        pass

    def estimated_remaining_instances(self):
        return (self.num_samples - self.instance_index)

    def has_more_instances(self):
        return (self.num_samples - self.instance_index > 0)

    def next_instance(self, batch_size=1):
        self.instance_index += batch_size
        # self.current_instance_x = self.X[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        # self.current_instance_y = self.y[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        try:

            self.current_instance_x = self.X[self.instance_index - batch_size:self.instance_index,:]
            self.current_instance_y = self.y[self.instance_index - batch_size:self.instance_index,:]
            if self.num_target_tasks < 2:
                self.current_instance_y = self.current_instance_y.flatten()

            '''
            if self.class_last:
                self.current_instance_x = self.X.iloc[self.instance_index - 1:self.instance_index + batch_size - 1, :].values
                self.current_instance_y = self.y.iloc[self.instance_index - 1:self.instance_index + batch_size - 1, :].values
                if self.num_classification_tasks < 2:
                    self.current_instance_y = self.current_instance_y.flatten()
            else:
                self.current_instance_x = self.X.iloc[self.instance_index - 1:self.instance_index + batch_size - 1,
                                          :].values
                self.current_instance_y = self.y.iloc[self.instance_index - 1:self.instance_index + batch_size - 1,
                                          :].values
                if self.num_classification_tasks < 2:
                    self.current_instance_y = self.current_instance_y.flatten()
            '''

        except IndexError:
            self.current_instance_x = None
            self.current_instance_y = None
        return (self.current_instance_x, self.current_instance_y)

    def is_restartable(self):
        return True

    def restart(self):
        pass

    def has_more_mini_batch(self):
        pass

    def get_num_nominal_attributes(self):
        pass

    def get_num_numerical_attributes(self):
        pass

    def get_num_values_per_nominal_attribute(self):
        pass

    def get_num_attributes(self):
        return self.num_features

    def get_num_targets(self):
        return self.num_target_tasks

    def get_attributes_header(self):
        pass

    def get_classes_header(self):
        pass

    def get_last_instance(self):
        return self.current_instance_x, self.current_instance_y

    def prepare_for_use(self):
        pass

    def get_plot_name(self):
        return 'Multilabel dataset'

    def get_classes(self):
        return np.unique(self.y)

    def get_class_type(self):
        return 'stream'

    def get_info(self):
        return 'Multilabel generator'

    def get_num_targeting_tasks(self):
        return self.num_target_tasks
