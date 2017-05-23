__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.instances.InstanceData import InstanceData
from skmultiflow.core.instances.InstanceHeader import InstanceHeader

class Instance:
    #def __init__(self, numAtt = None, numClasses = 1, w = -1, attHeader = None, attributes = None):
    def __init__(self, numAtt=None, numClasses=1, w=-1, attributes = None):
        ''' 
            By default we have 1 class
            The headers are stored as a python list
            attributes and classes are stores as numpy array     
        '''
        #if numAtt is None:
        #    self.numAttributes = len(attHeader) - numClasses
        #else:
        self.numAttributes = numAtt

        self.numClasses = numClasses
        self.weight = w
        #print(attributes[0:self.numAttributes])
        #print(attributes[self.numAttributes:])
        self.instanceData = InstanceData(attributes[0:self.numAttributes],
                                         attributes[self.numAttributes:])
        #self.instanceHeader = InstanceHeader(attHeader)

    def getAttribute(self, attIndex = -1):
        if (attIndex > -1):
            return self.instanceData.getAttributeAt[attIndex]
        else:
            return None

    def setClassValue(self, classVal = -1):
        self.instanceData.setClassValue(classVal)

    def getLabelSet(self):
        return True

    def getNumAttributes(self):
        return self.numAttributes

    def getNumClasses(self):
        return self.numClasses

    def getWeight(self):
        return self.weight

    def toString(self):
        #print(str(self.numAttributes-1))
        #string = "Attributes:\n"
        #for i in range(self.numAttributes):
        #    string += str(self.instanceHeader.getHeaderLabelAt(i)) + ": " + str(self.instanceData.getAttributeAt(i)) + "\n"

        #string += "\nClass:\n"
        #for i in range(self.numClasses):
        #string += str(self.instanceHeader.getHeaderLabelAt(self.numAttributes)) + ": " + str(self.instanceData.getClass()) + "\n"

        print("Attributes: ")
        print(self.instanceData.attributes)
        print("Class: ")
        print(self.instanceData.classes)