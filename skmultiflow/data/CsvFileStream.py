__author__ = 'Guilherme Matsumoto'

from skmultiflow.data import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
import pandas as pd

class CsvFileStream(BaseInstanceStream.BaseInstanceStream):
    '''
        CSV File Stream
        -------------------------------------------
        Generates a stream based on the data from a file
        
        Parser parameters
        ---------------------------------------------
        -f: CSV file to load
    '''
    def __init__(self, fileOpt, numClasses = 2):
        super().__init__()
        # default values
        if fileOpt.fileType is "CSV":
            self.read_function = pd.read_csv
        else:
            raise ValueError('Unsupported format: ', fileOpt.fileType)
        self.fileName = None
        self.instanceIndex = 0
        self.instanceLength = 0
        self.X = None
        self.y = None
        self.currentInstanceX = None
        self.currentInstanceY = None
        self.numClasses = numClasses
        self.numClasses = 0
        self.numNumericalAttributes = 0
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.attributesHeader = None
        self.classesHeader = None
        self.configure(fileOpt, numClasses, 0)

    def configure(self, fileOpt, numClasses, index = -1):
        '''
        __init__(self, fileName, index)
        
        Parameters
        ----------------------------------------
        fileName : string
                   Name of the file
        index : int
                Class index parameter
        '''
        self.fileName = fileOpt.getFileName()
        self.instanceIndex = index
        self.instances = None
        self.instanceLength = 0
        self.currentInstanceX = None
        self.currentInstanceY = None
        self.numClasses = numClasses
        self.X = None
        self.y = None




    def prepareForUse(self):
        self.restart()


    def restart(self):
        '''
        restart(self)
        ----------------------------------------
        Read the file and set object attributes
        '''

        try:
            instanceAux = self.read_function(self.fileName)
            self.instanceLength = len(instanceAux.index)
            self.numAttributes = len(instanceAux.columns) - 1
            labels = instanceAux.columns.values.tolist()
            self.X = instanceAux.iloc[:, 0:(len(labels)-1)]
            self.y = instanceAux.iloc[:, (len(labels)-1):]
            self.attributesHeader = labels[0:(len(labels)-1)]
            self.classesHeader = labels[(len(labels)-1):]
            self.instanceIndex = 0
        except IOError:
            print("CSV file reading failed. Please verify the file format.")
        pass

    def isRestartable(self):
        return True

    def nextInstance(self, batchSize = 1):

        self.instanceIndex += 1
        #self.currentInstanceX = self.X[self.instanceIndex-1:self.instanceIndex+batchSize-1].values[0]
        #self.currentInstanceY = self.y[self.instanceIndex-1:self.instanceIndex+batchSize-1].values[0]
        self.currentInstanceX = self.X.iloc[self.instanceIndex-1:self.instanceIndex+batchSize-1, : ].values
        self.currentInstanceY = self.y.iloc[self.instanceIndex-1:self.instanceIndex+batchSize-1, : ].values.flatten()
        return (self.currentInstanceX, self.currentInstanceY)

    def hasMoreInstances(self):
        return ((self.instanceLength - self.instanceIndex) > 0)

    def estimatedRemainingInstances(self):
        return (self.instanceLength - self.instanceIndex)

    def printDF(self):
        print(self.X)
        print(self.Y)

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
        return (self.currentInstanceX, self.currentInstanceY)
