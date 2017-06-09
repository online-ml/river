__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.generators.RandomRBFGenerator import RandomRBFGenerator
import numpy as np


class RandomRBFGeneratorDrift(RandomRBFGenerator):
    def __init__(self, modelSeed=21, instanceSeed=5, numClasses=2, numAtt=10, numCentroids=50, changeSpeed=0.0, numDriftCentroids=50):
        super().__init__(modelSeed, instanceSeed, numClasses, numAtt, numCentroids)
        #default values
        self.changeSpeed = changeSpeed
        self.numDriftCentroids = numDriftCentroids
        self.centroidSpeed = None
        pass

    def nextInstance(self, batchSize=1):
        data = np.zeros([batchSize, self.numNumericalAttributes + 1])
        for k in range(batchSize):
            len = self.numDriftCentroids
            if (len > self.numCentroids):
                len = self.numCentroids

            for i in range(len):
                for j in range(self.numNumericalAttributes):
                    self.centroids[i].centre[j] += self.centroidSpeed[i][j]*self.changeSpeed

                    if ((self.centroids[i].centre[j] > 1) | (self.centroids[i].centre[j] < 0)):
                        self.centroids[i].centre[j] = 1 if (self.centroids[i].centre[j] > 1) else 0
                        self.centroidSpeed[i][j] = -self.centroidSpeed[i][j]
            X, y = super().nextInstance(1)
            data[k, :] = np.concatenate((X[0], y[0]))
        return (data[:, :self.numNumericalAttributes], data[:, self.numNumericalAttributes:])

    def generateCentroids(self):
        super().generateCentroids()
        modelRandom = np.random
        modelRandom.seed(self.modelSeed)
        len = self.numDriftCentroids
        self.centroidSpeed = []
        if (len > self.numCentroids):
            len = self.numCentroids

        for i in range(len):
            randSpeed = []
            normSpeed = 0.0
            for j in range(self.numNumericalAttributes):
                randSpeed.append(modelRandom.rand())
                normSpeed += randSpeed[j]*randSpeed[j]
            normSpeed = np.sqrt(normSpeed)
            #print(randSpeed)
            for j in range(self.numNumericalAttributes):
                randSpeed[j] /= normSpeed
            self.centroidSpeed.append(randSpeed)

    def prepareForUse(self):
        self.restart()

    def restart(self):
        self.generateCentroids()
        self.instanceRandom.seed(self.instanceSeed)

if __name__ == '__main__':
    stream = RandomRBFGeneratorDrift(changeSpeed=0.02, numDriftCentroids=50)
    stream.prepareForUse()

    X, y = stream.nextInstance(4)
    print(X)
    print(y)