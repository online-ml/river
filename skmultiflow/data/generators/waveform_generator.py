__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.base_object import BaseObject
import numpy as np
from timeit import default_timer as timer


class WaveformGenerator(BaseInstanceStream, BaseObject):
    """ WaveformGenerator
    
    Generates instances with 21 numeric attributes and 3 targets, based 
    on a random differentiation of some base waveforms. Supports noise 
    addition, but in this case the generator will have 40 attribute 
    instances
     
    Parameters
    ----------
    seed: int
        Seed for random generation of instances (Default: 23)
    add_noise: bool
        Add noise (Default: False)
        
    """
    
    NUM_CLASSES = 3
    NUM_BASE_ATTRIBUTES = 21
    TOTAL_ATTRIBUTES_INCLUDING_NOISE = 40
    H_FUNCTION = np.array([[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
                           [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]])

    def __init__(self, seed=23, add_noise=False):
        super().__init__()
        # default values
        self.random_seed = 1
        self.add_noise = False
        self.num_numerical_attributes = self.NUM_BASE_ATTRIBUTES
        self.num_classes = self.NUM_CLASSES
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.attributes_header = None
        self.classes_header = None
        self.current_instance_x = None
        self.current_instance_y = None
        self.instance_length = 0
        self.num_attributes = 0
        self.instance_index = 0

        self.__configure(seed, add_noise)
        pass

    def __configure(self, seed, add_noise):
        self.random_seed = seed if seed is not None else 23
        self.add_noise = add_noise if add_noise is not None else False

        self.instance_length = 100000
        self.num_attributes = self.TOTAL_ATTRIBUTES_INCLUDING_NOISE if self.add_noise else self.NUM_BASE_ATTRIBUTES
        self.num_classes = self.NUM_CLASSES
        self.attributes_header = []
        self.classes_header = ["class"]
        for i in range(self.num_attributes):
            self.attributes_header.append("att" + str(i))


    def prepare_for_use(self):
        self.restart()

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size = 1):
        """ next_instance
        
        An instance is generated based on the parameters passed. If noise 
        is included the total number of attributes will be 40, if it's not 
        included there will be 21 attributes. In both cases there is one 
        classification task, which chooses one between three labels.
        
        After the number of attributes is chosen, the algorithm will randomly
        choose one of the hard coded waveforms, as well as random multipliers. 
        For each attribute, the actual value generated will be a a combination 
        of the hard coded functions, with the multipliers and a random value.
        
        Furthermore, if noise is added the attributes from 21 to 40 will be 
        replaced with a random normal value.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
            
        Returns
        -------
        Return a tuple with the features matrix and the labels matrix 
        for the batch_size samples that were requested.
        
        """
        if self.has_noise():
            data = np.zeros([batch_size, self.TOTAL_ATTRIBUTES_INCLUDING_NOISE + 1])
        else:
            data = np.zeros([batch_size, self.NUM_BASE_ATTRIBUTES + 1])

        for j in range (batch_size):
            self.instance_index += 1
            waveform = np.random.randint(0, self.NUM_CLASSES)
            choice_a = 1 if (waveform == 2) else 0
            choice_b = 1 if (waveform == 0) else 2
            multiplier_a = np.random.rand()
            multiplier_b = 1.0 - multiplier_a

            for i in range(self.NUM_BASE_ATTRIBUTES):
                data[j,i] = multiplier_a*self.H_FUNCTION[choice_a][i] \
                            + multiplier_b*self.H_FUNCTION[choice_b][i] \
                            + np.random.normal()

            if self.has_noise():
                for i in range(self.NUM_BASE_ATTRIBUTES,self.TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j,i] = np.random.normal()

            data[j, data[j].size-1] = waveform
            self.current_instance_x = data[j, :self.num_attributes]
            self.current_instance_y = data[j, self.num_attributes:]

        return (data[:, :self.num_attributes], np.ravel(data[:, self.num_attributes:]))

    def is_restartable(self):
        return True

    def restart(self):
        np.random.seed(self.random_seed)
        self.instance_index = 0
        pass

    def has_more_mini_batch(self):
        return True

    def has_noise(self):
        return self.add_noise

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.num_numerical_attributes + self.num_nominal_attributes * self.num_values_per_nominal_att

    def get_num_targets(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.classes_header

    def get_last_instance(self):
        return (self.current_instance_x, self.current_instance_y)

    def get_plot_name(self):
        return "Waveform Generator - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def get_info(self):
        add_noise = 'True' if self.add_noise else 'False'
        return 'Waveform Generator: num_classes: ' + str(self.num_classes) + \
               '  -  num_numerical_attributes: ' + str(self.num_numerical_attributes) + \
               '  -  num_nominal_attributes: ' + str(self.num_nominal_attributes) + \
               '  -  add_noise: ' + add_noise + \
               '  -  random_seed: ' + str(self.random_seed)

    def get_num_targeting_tasks(self):
        return 1

def demo():
    wfg = WaveformGenerator()
    wfg.prepare_for_use()
    print(wfg.get_class_type())
    i = 0
    start = timer()
    oi = np.zeros([4])
    oi.put(0, 3)
    oi.put(2, 1.3)
    oi.put(3, 9)
    oi.put(1, 4.5)
    print(oi)
    print(oi[3])
    end = timer()
    print("Generation time: " + str(end-start))

if __name__ == '__main__':
    demo()