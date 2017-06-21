__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.generators.RandomRBFGenerator import RandomRBFGenerator
from skmultiflow.core.BaseObject import BaseObject
import numpy as np


class RandomRBFGeneratorDrift(RandomRBFGenerator, BaseObject):
    def __init__(self, model_seed=21, instance_seed=5, num_classes=2, num_att=10, num_centroids=50, change_speed=0.0, num_drift_centroids=50):
        super().__init__(model_seed, instance_seed, num_classes, num_att, num_centroids)
        #default values
        self.change_speed = change_speed
        self.num_drift_centroids = num_drift_centroids
        self.centroid_speed = None
        pass

    def next_instance(self, batch_size=1):
        data = np.zeros([batch_size, self.num_numerical_attributes + 1])
        for k in range(batch_size):
            len = self.num_drift_centroids
            if (len > self.num_centroids):
                len = self.num_centroids

            for i in range(len):
                for j in range(self.num_numerical_attributes):
                    self.centroids[i].centre[j] += self.centroid_speed[i][j] * self.change_speed

                    if ((self.centroids[i].centre[j] > 1) | (self.centroids[i].centre[j] < 0)):
                        self.centroids[i].centre[j] = 1 if (self.centroids[i].centre[j] > 1) else 0
                        self.centroid_speed[i][j] = -self.centroid_speed[i][j]
            X, y = super().next_instance(1)
            data[k, :] = np.concatenate((X[0], y[0]))
        return (data[:, :self.num_numerical_attributes], data[:, self.num_numerical_attributes:])

    def generate_centroids(self):
        super().generate_centroids()
        model_random = np.random
        model_random.seed(self.model_seed)
        len = self.num_drift_centroids
        self.centroid_speed = []
        if (len > self.num_centroids):
            len = self.num_centroids

        for i in range(len):
            rand_speed = []
            norm_speed = 0.0
            for j in range(self.num_numerical_attributes):
                rand_speed.append(model_random.rand())
                norm_speed += rand_speed[j]*rand_speed[j]
            norm_speed = np.sqrt(norm_speed)
            #print(rand_speed)
            for j in range(self.num_numerical_attributes):
                rand_speed[j] /= norm_speed
            self.centroid_speed.append(rand_speed)

    def prepare_for_use(self):
        self.restart()

    def restart(self):
        self.generate_centroids()
        self.instance_random.seed(self.instance_seed)

    def get_info(self):
        pass

if __name__ == '__main__':
    stream = RandomRBFGeneratorDrift(change_speed=0.02, num_drift_centroids=50)
    stream.prepare_for_use()

    X, y = stream.next_instance(4)
    print(X)
    print(y)