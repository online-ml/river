__author__ = 'Guilherme Matsumoto'

from skmultiflow.data import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
import pandas as pd

class CsvFileStream(BaseInstanceStream.BaseInstanceStream):
    '''
        CSV File Stream
        -------------------------------------------
        Generates a stream based on the data from an ARFF file
        
        Parser parameters
        ---------------------------------------------
        -f: CSV file to load
    '''
    def __init__(self, fileOpt, numAtt = 0):
        super().__init__()
        # default values
        self.arffFileName = None
        self.instanceIndex = 0
        self.instances = None
        self.instanceLength = 0
        self.currentInstance = None
        self.numAttributes = numAtt
        self.numClasses = 0
        self.numNumericalAttributes = 0
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.attributesHeader = None
        self.classesHeader = None
        self.configure(fileOpt, 0, numAtt)

    def configure(self, fileOpt, index = -1, numAtt = 0):
        '''
        __init__(self, fileName, index)
        
        Parameters
        ----------------------------------------
        fileName : string
                   Name of the ARFF file
        index : int
                Class index parameter
        '''
        self.arffFileName = fileOpt.getFileName()
        self.instanceIndex = index
        self.instances = None
        self.instanceLength = 0
        self.currentInstance = None
        self.numAttributes = numAtt
        self.numClasses = 0




    def prepareForUse(self):
        self.restart()


    def restart(self):
        '''
        restart(self)
        ----------------------------------------
        Read the file and set object attributes
        '''

        try:
            self.instances = pd.read_csv(self.arffFileName)
            self.instanceLength = len(self.instances.index)
            self.numClasses = len(self.instances.columns) - self.numAttributes
            labels = self.instances.columns.values.tolist()
            self.attributesHeader = labels[0:(len(labels)-1)]
            self.classesHeader = labels[(len(labels)-1):]
        except IOError:
            print("CSV file reading failed. Please verify the file format.")
        pass

    def isRestartable(self):
        return True

    def nextInstance(self):
        self.instanceIndex += 1
        self.currentInstance = Instance(self.numAttributes,
                                        self.numClasses, -1,
                                        self.instances[self.instanceIndex-1:self.instanceIndex].values[0])
        return self.currentInstance

    def hasMoreInstances(self):
        return ((self.instanceLength - self.instanceIndex) > 0)

    def estimatedRemainingInstances(self):
        return (self.instanceLength - self.instanceIndex)

    def printDF(self):
        print(self.instances)

    def getInstancesLength(self):
        return self.instanceLength

    def getNumAttributes(self):
        return self.numAttributes

    def nextInstanceMiniBatch(self):
        pass

    def hasMoreMiniBatch(self):
        pass

    def getNumNominalAttributes(self):
        return self.numNominalAttributes

    def getNumNumericalAttributes(self):
        return self.numNumericalAttributes

    def getNumValuesPerNominalAttribute(self):
        return self.numValuesPerNominalAtt

    def getNumClasses(self):
        return self.numClasses

    def getAttributesHeader(self):
        return self.attributesHeader

    def getClassesHeader(self):
        return self.classesHeader

    def getLastInstance(self):
        return self.currentInstance

    def getNumLabels(self):
        return self.numLabels