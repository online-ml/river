__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.BaseInstanceStream import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
from skmultiflow.core.instances.InstanceHeader import InstanceHeader
import numpy as np


from timeit import default_timer as timer

class WaveformGenerator(BaseInstanceStream):
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

    def __init__(self, optList = None):
        super().__init__()
        # default values
        self.randomSeed = 1
        self.addNoise = False
        self.numNumericalAttributes = self.NUM_BASE_ATTRIBUTES
        self.numClasses = self.NUM_CLASSES
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.attributesHeader = None
        self.classesHeader = None
        self.configure(optList)
        pass

    def configure(self, optList):
        if optList is not None:
            for i in range(len(optList)):
                opt, arg = optList[i]
                if opt in ("-i"):
                    self.randomSeed = int(arg)
                elif opt in ("-n"):
                    self.addNoise = True

        self.instanceLength = 100000
        self.numAttributes = self.TOTAL_ATTRIBUTES_INCLUDING_NOISE if self.addNoise else self.NUM_BASE_ATTRIBUTES
        self.numClasses = self.NUM_CLASSES
        self.attributesHeader = []
        self.classesHeader = ["class"]
        for i in range(self.numAttributes):
            self.attributesHeader.append("att" + str(i))

        #for i in range(self.numClasses):
        self.classesHeader = InstanceHeader(self.classesHeader)
        #print(self.attributesHeader)
        #print(self.classesHeader)

    def prepareForUse(self):
        self.restart()
        pass

    def estimatedRemainingInstances(self):
        return -1

    def hasMoreInstances(self):
        return True

    def nextInstance(self):
        self.instanceIndex += 1
        waveform = np.random.randint(0, self.NUM_CLASSES)
        choiceA = 1 if (waveform == 2) else 0
        choiceB = 1 if (waveform == 0) else 2
        multiplierA = np.random.rand()
        multiplierB = 1.0 - multiplierA
        if self.hasNoise():
            data = np.zeros([self.TOTAL_ATTRIBUTES_INCLUDING_NOISE+1])
        else:
            data = np.zeros([self.NUM_BASE_ATTRIBUTES+1])
        for i in range(self.NUM_BASE_ATTRIBUTES):
            data.put(i, multiplierA*self.H_FUNCTION[choiceA][i]
                        + multiplierB*self.H_FUNCTION[choiceB][i]
                        + np.random.normal())
        if self.hasNoise():
            for i in range(self.NUM_BASE_ATTRIBUTES,self.TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                data.put(i, np.random.normal())

        data.put(data.size - 1, waveform)
        self.currentInstance = Instance(self.numAttributes, self.numClasses, -1, data)
        self.currentInstance.setClassValue(waveform)
        return self.currentInstance

    def isRestartable(self):
        return True

    def restart(self):
        np.random.seed(self.randomSeed)
        self.instanceIndex = 0
        pass

    def nextInstanceMiniBatch(self):
        pass

    def hasMoreMiniBatch(self):
        pass

    def hasNoise(self):
        return self.addNoise

    def getNumNominalAttributes(self):
        return self.numNominalAttributes

    def getNumNumericalAttributes(self):
        return self.numNumericalAttributes

    def getNumValuesPerNominalAttribute(self):
        return self.numValuesPerNominalAtt

    def getNumAttributes(self):
        return self.numNumericalAttributes + self.numNominalAttributes*self.numValuesPerNominalAtt

    def getNumClasses(self):
        return self.numClasses

    def getAttributesHeader(self):
        return self.attributesHeader

    def getClassesHeader(self):
        return self.classesHeader

    def getLastInstance(self):
        return self.currentInstance

    def getNumLabels(self):
        pass

def demo():
    wfg = WaveformGenerator()
    wfg.prepareForUse()

    i = 0
    start = timer()
    #while(wfg.hasMoreInstances()):
    #for i in range(20000):
        #o = wfg.nextInstance()
        #o.toString()
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