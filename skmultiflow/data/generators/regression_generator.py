__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from sklearn.datasets import make_regression
import numpy as np


class RegressionGenerator(BaseInstanceStream):
    def __init__(self, n_samples=40000, n_features=100, n_informative=10, n_targets=1):
        super().__init__()
        self.X = None
        self.y = None
        self.num_samples = 0
        self.num_features = 0
        self.num_target_tasks = 0
        self.num_informative = 0
        self.instance_index = 0
        self.current_instance_y = None
        self.current_instance_x = None
        self.configure(n_samples, n_features, n_informative, n_targets)

    def configure(self, n_samples, n_features, n_informative, n_targets):
        self.X, self.y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_targets=n_targets)
        self.y.resize((self.y.size, n_targets))
        self.num_samples = n_samples
        self.num_features = n_features
        self.num_target_tasks = n_targets
        self.num_informative = n_informative

    def estimated_remaining_instances(self):
        return (self.num_samples - self.instance_index)

    def has_more_instances(self):
        return (self.num_samples - self.instance_index > 0)

    def next_instance(self, batch_size=1):
        self.instance_index += batch_size
        # self.current_instance_x = self.X[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        # self.current_instance_y = self.y[self.instance_index-1:self.instance_index+batchSize-1].values[0]
        try:

            self.current_instance_x = self.X[self.instance_index - batch_size:self.instance_index, :]
            self.current_instance_y = self.y[self.instance_index - batch_size:self.instance_index, :]
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
        pass

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
        return 'Regression dataset'

    def get_classes(self):
        pass

    def get_class_type(self):
        return 'stream'

    def get_info(self):
        return 'Regression generator'

    def get_num_targeting_tasks(self):
        return self.num_target_tasks
