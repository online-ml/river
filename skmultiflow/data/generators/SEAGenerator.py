__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.BaseInstanceStream import BaseInstanceStream
import numpy as np


class SEAGenerator(BaseInstanceStream):
    def __init__(self, classificationFunction = 0, instanceSeed = 42, balanceClasses = False, noisePercentage = 0.0):
        super().__init__()

        #classification functions to use
        self.classificationFunctions = [self.classificationFunctionZero, self.classificationFunctionOne,
                                        self.classificationFunctionTwo, self.classificationFunctionThree]

        #default values
        self.numNumericalAttributes = 3
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.numClasses = 2
        self.currentInstanceX = None
        self.currentInstanceY = None

        self.configure(classificationFunction, instanceSeed, balanceClasses, noisePercentage)
        pass

    def configure(self, classificationFunction, instanceSeed, balanceClasses, noisePercentage):
        self.classificationFunctionIndex = classificationFunction
        self.instanceSeed = instanceSeed
        self.balanceClasses = balanceClasses
        self.noisePercentage = noisePercentage
        self.instanceRandom = np.random
        self.instanceRandom.seed(self.instanceSeed)
        self.nextClassShouldBeZero = False

        self.classHeader = ["class"]
        self.attributesHeader = []
        for i in range(self.numNumericalAttributes):
            self.attributesHeader.append("NumAtt" + str(i))
        pass

    def estimatedRemainingInstances(self):
        return -1

    def hasMoreInstances(self):
        return True

    def nextInstance(self, batchSize = 1):
        data = np.zeros([batchSize, self.numNumericalAttributes + 1])

        for j in range (batchSize):
            att1 = att2 = att3 = 0.0
            group = 0
            desiredClassFound = False
            while not desiredClassFound:
                att1 = 10*self.instanceRandom.rand()
                att2 = 10*self.instanceRandom.rand()
                att3 = 10*self.instanceRandom.rand()
                group = self.classificationFunctions[self.classificationFunctionIndex](att1, att2, att3)

                if not self.balanceClasses:
                    desiredClassFound = True
                else:
                    if ((self.nextClassShouldBeZero & (group == 0)) | ((not self.nextClassShouldBeZero) & (group == 1))):
                        desiredClassFound = True
                        self.nextClassShouldBeZero = not self.nextClassShouldBeZero

            if ((0.01 + self.instanceRandom.rand() <= self.noisePercentage)):
                #print("noise " + str(j))
                group = 1 if (group == 0) else 0

            data[j, 0] = att1
            data[j, 1] = att2
            data[j, 2] = att3
            data[j, 3] = group

            self.currentInstanceX = data[j, :self.numNumericalAttributes]
            self.currentInstanceY = data[j, self.numNumericalAttributes:]

        return (data[:, :self.numNumericalAttributes], data[:, self.numNumericalAttributes:])

    def prepareForUse(self):
        self.restart()

    def isRestartable(self):
        return True

    def restart(self):
        self.instanceRandom.seed(self.instanceSeed)
        self.nextClassShouldBeZero = False
        pass

    def hasMoreMiniBatch(self):
        return True

    def getNumNominalAttributes(self):
        return self.numNominalAttributes

    def getNumNumericalAttributes(self):
        return self.numNumericalAttributes

    def getNumValuesPerNominalAttribute(self):
        return self.numValuesPerNominalAtt

    def getNumAttributes(self):
        return self.numNumericalAttributes + (self.numNominalAttributes*self.numValuesPerNominalAtt)

    def getNumClasses(self):
        return self.numClasses

    def getAttributesHeader(self):
        return self.attributesHeader

    def getClassesHeader(self):
        return self.classHeader

    def getLastInstance(self):
        return (self.currentInstanceX, self.currentInstanceY)

    def classificationFunctionZero(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 8) else 1

    def classificationFunctionOne(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 9) else 1

    def classificationFunctionTwo(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 7) else 1

    def classificationFunctionThree(self, att1, att2, att3):
        return 0 if (att1 + att2 <= 9.5) else 1

    def getPlotName(self):
        return "SEA Generator - " + str(self.numClasses) + " class labels"

    def getClasses(self):
        c = []
        for i in range(self.numClasses):
            c.append(i)
        return c

if __name__ == "__main__":
    sg = SEAGenerator(classificationFunction=3, noisePercentage=0.2)

    X, y = sg.nextInstance(10)
    print(X)
    print(y)
