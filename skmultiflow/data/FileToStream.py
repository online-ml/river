__author__ = 'Jacob Montiel'

from skmultiflow.data import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
import pandas as pd
import numpy as np

class FileToStream(BaseInstanceStream.BaseInstanceStream):
    '''
        Converts a data file into a data stream
        -------------------------------------------
        Generates a stream from a file
        
        Parser parameters
        ---------------------------------------------
        -f: file
    '''
    def __init__(self, fileOpt, numClasses = 2):
        super().__init__()
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
        if fileOpt.fileType is "CSV":
            self.read_function = pd.read_csv
        else:
            raise ValueError('Unsupported format: ', fileOpt.fileType)
        self.instanceIndex = 0
        self.instances = None
        self.numInstances = 0
        self.currentInstance = None
        self.numAttributes = numAtt
        self.numClasses = 0
        self.numNumericalAttributes = 0
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.attributesHeader = None
        self.classesHeader = None

    def prepareForUse(self):
        self.restart()

    def restart(self):
        '''
        restart(self)
        ----------------------------------------
        Read the file and set object attributes
        '''
        if not self.instances:
            try:
                self.instances = self.read_function(self.fileName)
                self.numInstances = self.instances.shape[0]
                self.numClasses = self.instances.shape[1] - self.numAttributes
                labels = self.instances.columns.values.tolist()
                self.attributesHeader = labels[0:(len(labels)-self.numClasses)]
                self.classesHeader = labels[(len(labels)-self.numClasses):]
            except IOError:
                print("File reading failed.")
        else:
            self.instanceIndex = 0
        pass

    def isRestartable(self):
        return True

    def nextInstance(self, batchSize = 1):
        self.currentInstance = Instance(self.numAttributes,
                                        self.numClasses, -1,
                                        self.instances.iloc[self.instanceIndex:self.instanceIndex+1].values[0])
        self.instanceIndex += 1
        return self.currentInstance

    def hasMoreInstances(self):
        return ((self.numInstances - self.instanceIndex) > 0)

    def estimatedRemainingInstances(self):
        return (self.numInstances - self.instanceIndex)

    def printDF(self):
        print(self.instances)

    def getInstancesLength(self):
        return self.numInstances

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

    def getPlotName(self):
        return "File Stream - " + str(self.numClasses) + " class labels"

    def getClasses(self):
        c = np.unique(self.instances[:, self.numAttributes:])
        c = []
        for i in range(self.numClasses):
            c.append(i)
        return c