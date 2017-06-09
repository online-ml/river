__author__ = 'Guilherme Matsumoto'

from numpy import ndarray


class InstanceData:
    '''
        Instance Data class
        -----------------------------
        Stores data from an instance, both the features and the classes
    '''
    def __init__(self, attributesList = None, labelsList = None):
        #self.attributes = attributesList.values.tolist()
        #self.labels = labelsList.values.tolist()
        self.attributes = attributesList
        self.classes = labelsList
        #print(self.attributes)
        #print(self.classes)
        #problem is fucking here
        #print(str(len(self.attributes)) + " " + str(len(self.labels)))
        pass

    def get_attribute_at(self, attIndex = -1):
        return self.attributes[attIndex] if ((attIndex > -1) & (attIndex < len(self.attributes))) else None


    def get_class(self):
        return self.classes[0]

    def set_class_value(self, classVal = -1):
        self.classes[0] = classVal
        pass