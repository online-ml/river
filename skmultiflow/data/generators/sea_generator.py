__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.base_object import BaseObject
import numpy as np


class SEAGenerator(BaseInstanceStream, BaseObject):
    def __init__(self, classification_function = 0, instance_seed = 42, balance_classes = False, noise_percentage = 0.0):
        super().__init__()

        #classification functions to use
        self.classification_functions = [self.classification_function_zero, self.classification_function_one,
                                         self.classification_function_two, self.classification_function_three]

        #default values
        self.num_numerical_attributes = 3
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.num_classes = 2
        self.current_instance_x = None
        self.current_instance_y = None
        self.classification_function_index = None
        self.instance_seed = None
        self.balance_classes = None
        self.noise_percentage = None
        self.instance_random = None
        self.next_class_should_be_zero = False

        self.class_header = None
        self.attributes_header = None

        self.configure(classification_function, instance_seed, balance_classes, noise_percentage)
        pass

    def configure(self, classification_function, instance_seed, balance_classes, noise_percentage):
        self.classification_function_index = classification_function
        self.instance_seed = instance_seed
        self.balance_classes = balance_classes
        self.noise_percentage = noise_percentage
        self.instance_random = np.random
        self.instance_random.seed(self.instance_seed)
        self.next_class_should_be_zero = False

        self.class_header = ["class"]
        self.attributes_header = []
        for i in range(self.num_numerical_attributes):
            self.attributes_header.append("NumAtt" + str(i))
        pass

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size = 1):
        data = np.zeros([batch_size, self.num_numerical_attributes + 1])

        for j in range (batch_size):
            att1 = att2 = att3 = 0.0
            group = 0
            desired_class_found = False
            while not desired_class_found:
                att1 = 10*self.instance_random.rand()
                att2 = 10*self.instance_random.rand()
                att3 = 10*self.instance_random.rand()
                group = self.classification_functions[self.classification_function_index](att1, att2, att3)

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if ((self.next_class_should_be_zero & (group == 0)) | ((not self.next_class_should_be_zero) & (group == 1))):
                        desired_class_found = True
                        self.next_class_should_be_zero = not self.next_class_should_be_zero

            if ((0.01 + self.instance_random.rand() <= self.noise_percentage)):
                #print("noise " + str(j))
                group = 1 if (group == 0) else 0

            data[j, 0] = att1
            data[j, 1] = att2
            data[j, 2] = att3
            data[j, 3] = group

            self.current_instance_x = data[j, :self.num_numerical_attributes]
            self.current_instance_y = data[j, self.num_numerical_attributes:]

        return (data[:, :self.num_numerical_attributes], data[:, self.num_numerical_attributes:])

    def prepare_for_use(self):
        self.restart()

    def is_restartable(self):
        return True

    def restart(self):
        self.instance_random.seed(self.instance_seed)
        self.next_class_should_be_zero = False
        pass

    def has_more_mini_batch(self):
        return True

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)

    def get_num_classes(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.class_header

    def get_last_instance(self):
        return (self.current_instance_x, self.current_instance_y)

    def classification_function_zero(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 8) else 1

    def classification_function_one(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 9) else 1

    def classification_function_two(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 7) else 1

    def classification_function_three(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 9.5) else 1

    def get_plot_name(self):
        return "SEA Generator - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def get_info(self):
        pass

if __name__ == "__main__":
    sg = SEAGenerator(classification_function=3, noise_percentage=0.2)

    X, y = sg.next_instance(10)
    print(X)
    print(y)
