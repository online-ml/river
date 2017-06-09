__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.BaseInstanceStream import BaseInstanceStream
from skmultiflow.core.utils import PseudoRandomProcesses as prp
import numpy as np


class RandomRBFGenerator(BaseInstanceStream):
    def __init__(self, modelSeed = 21, instanceSeed = 5, numClasses = 2, numAtt = 10, numCentroids = 50):
        super().__init__()

        #default values
        self.numNumericalAttributes = 10
        self.numNominalAttributes = 0
        self.numValuesPerNominalAtt = 0
        self.currentInstanceX = None
        self.currentInstanceY = None
        self.modelSeed = 21
        self.instanceSeed = 5
        self.numClasses = 2
        self.numCentroids = 50
        self.centroids = None
        self.centroidWeights = None

        self.configure(modelSeed, instanceSeed, numClasses, numAtt, numCentroids)
        pass

    def configure(self, modelSeed, instanceSeed, numClasses, numAtt, numCentroids):
        self.modelSeed = modelSeed
        self.instanceSeed = instanceSeed
        self.numClasses = numClasses
        self.numNumericalAttributes = numAtt
        self.numCentroids = numCentroids
        self.instanceRandom = np.random
        self.instanceRandom.seed(self.instanceSeed)

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
        numAtts = self.numNumericalAttributes
        for j in range(batchSize):
            centroidAux = self.centroids[prp.randomIndexBasedOnWeights(self.centroidWeights, self.instanceRandom)]
            attVals = []
            magnitude = 0.0
            for i in range(numAtts):
                attVals.append((self.instanceRandom.rand()*2.0)-1.0)
                magnitude += attVals[i]*attVals[i]
            magnitude = np.sqrt(magnitude)
            desiredMag = self.instanceRandom.normal()*centroidAux.stdDev
            scale = desiredMag/magnitude
            for i in range(numAtts):
                data[j, i] = centroidAux.centre[i] + attVals[i]*scale
            data[j, numAtts] = centroidAux.classLabel
        return (data[:, :numAtts], data[:, numAtts:])

    def prepareForUse(self):
        self.restart()

    def isRestartable(self):
        return True

    def restart(self):
        self.generateCentroids()
        self.instanceRandom.seed(self.instanceSeed)
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

    def getPlotName(self):
        return "Random RBF Generator - " + str(self.numClasses) + " class labels"

    def getClasses(self):
        c = []
        for i in range(self.numClasses):
            c.append(i)
        return c

    def generateCentroids(self):
        modelRandom = np.random
        modelRandom.seed(self.modelSeed)
        self.centroids = []
        self.centroidWeights = []
        for i in range (self.numCentroids):
            self.centroids.append(Centroid())
            randCentre = []
            for j in range(self.numNumericalAttributes):
                randCentre.append(modelRandom.rand())
            self.centroids[i].centre = randCentre
            self.centroids[i].classLabel = modelRandom.randint(self.numClasses)
            self.centroids[i].stdDev = modelRandom.rand()
            self.centroidWeights.append(modelRandom.rand())
            pass

class Centroid:
    def __init__(self):
        self.centre = None
        self.classLabel = None
        self.stdDev = None


if __name__ == "__main__":
    rrbfg = RandomRBFGenerator()
    rrbfg.prepareForUse()
    for i in range(4):
        X, y = rrbfg.nextInstance(4)
        print(X)
        print(y)

