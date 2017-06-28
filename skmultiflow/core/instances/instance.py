__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.instances.instance_data import InstanceData
from skmultiflow.core.instances.instance_header import InstanceHeader
from skmultiflow.core.base_object import BaseObject

class Instance(BaseObject):
    #def __init__(self, numAtt = None, num_classes = 1, w = -1, attHeader = None, attributes = None):
    def __init__(self, numAtt=None, numClasses=1, w=-1, attributes = None):
        ''' 
            By default we have 1 class
            The headers are stored as a python list
            attributes and targets are stores as numpy array     
        '''
        #if numAtt is None:
        #    self.num_attributes = len(attHeader) - num_classes
        #else:
        super().__init__()
        self.num_attributes = numAtt

        self.num_classes = numClasses
        self.weight = w
        #print(attributes[0:self.num_attributes])
        #print(attributes[self.num_attributes:])
        self.instance_data = InstanceData(attributes[0:self.num_attributes],
                                         attributes[self.num_attributes:])
        #self.instanceHeader = InstanceHeader(attHeader)

    def get_attribute(self, attIndex = -1):
        if (attIndex > -1):
            return self.instance_data.get_attribute_at[attIndex]
        else:
            return None

    def set_class_value(self, classVal = -1):
        self.instance_data.set_class_value(classVal)

    def get_label_set(self):
        return True

    def get_num_attributes(self):
        return self.num_attributes

    def get_num_classes(self):
        return self.num_classes

    def get_weight(self):
        return self.weight

    def to_string(self):
        #print(str(self.num_attributes-1))
        #string = "Attributes:\n"
        #for i in range(self.num_attributes):
        #    string += str(self.instanceHeader.get_header_label_at(i)) + ": " + str(self.instanceData.get_attribute_at(i)) + "\n"

        #string += "\nClass:\n"
        #for i in range(self.num_classes):
        #string += str(self.instanceHeader.get_header_label_at(self.num_attributes)) + ": " + str(self.instanceData.get_class()) + "\n"

        print("Attributes: ")
        print(self.instance_data.attributes)
        print("Class: ")
        print(self.instance_data.classes)

    def get_class_type(self):
        return 'instance'

    def get_info(self):
        pass