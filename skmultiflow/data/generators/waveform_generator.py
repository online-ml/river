__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.instances.instance import Instance
from skmultiflow.core.instances.instance_header import InstanceHeader
from skmultiflow.core.base_object import BaseObject
import numpy as np
from timeit import default_timer as timer


class WaveformGenerator(BaseInstanceStream, BaseObject):
    '''
        WaveformGenerator
        ------------------------------------------
        Generates instances with 21 numeric attributes and 3 classes, based on a random differentiation of some base 
        wave forms. Supports noise addition, but in this case the generator will have 40 attribute instances
         
        Parser parameters
        ---------------------------------------------
        -i: Seed for random generation of instances (Default: 23)
        -n: Add noise (Default: False)
    '''
    NUM_CLASSES = 3
    NUM_BASE_ATTRIBUTES = 21
    TOTAL_ATTRIBUTES_INCLUDING_NOISE = 40
    H_FUNCTION = np.array([[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
                           [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]])

    def __init__(self, opt_list = None):
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

        self.configure(opt_list)
        pass

    def configure(self, opt_list):
        if opt_list is not None:
            for i in range(len(opt_list)):
                opt, arg = opt_list[i]
                if opt in ("-i"):
                    self.random_seed = int(arg)
                elif opt in ("-n"):
                    self.add_noise = True

        self.instance_length = 100000
        self.num_attributes = self.TOTAL_ATTRIBUTES_INCLUDING_NOISE if self.add_noise else self.NUM_BASE_ATTRIBUTES
        self.num_classes = self.NUM_CLASSES
        self.attributes_header = []
        self.classes_header = ["class"]
        for i in range(self.num_attributes):
            self.attributes_header.append("att" + str(i))


    def prepare_for_use(self):
        self.restart()
        pass

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size = 1):
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
            #print(data)
            #print(str(j))
            for i in range(self.NUM_BASE_ATTRIBUTES):
                #data.put(i, multiplier_a*self.H_FUNCTION[choice_a][i]
                            #+ multiplier_b*self.H_FUNCTION[choice_b][i]
                            #+ np.random.normal())
                data[j,i] = multiplier_a*self.H_FUNCTION[choice_a][i] \
                            + multiplier_b*self.H_FUNCTION[choice_b][i] \
                            + np.random.normal()
            #print(data)
            if self.has_noise():
                for i in range(self.NUM_BASE_ATTRIBUTES,self.TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    #data.put(i, np.random.normal())
                    data[j,i] = np.random.normal()
            #print(data)
            #data.put(data.size - 1, waveform)
            data[j, data[j].size-1] = waveform
            #self.current_instance_x = data[:self.num_attributes]
            #self.current_instance_y = data[self.num_attributes:]
            self.current_instance_x = data[j, :self.num_attributes]
            self.current_instance_y = data[j, self.num_attributes:]
            #print(self.current_instance_x)
            #print(self.current_instance_y)
            #print(data)
        #return (self.current_instance_x, self.current_instance_y)
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

    def get_num_classes(self):
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

def demo():
    wfg = WaveformGenerator()
    wfg.prepare_for_use()
    print(wfg.get_class_type())
    i = 0
    start = timer()
    #while(wfg.has_more_instances()):
    #for i in range(20000):
        #o = wfg.next_instance()
        #o.to_string()
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