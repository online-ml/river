__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.classification.core.driftdetection.adwin import ADWIN


def demo():
    adwin = ADWIN()
    size = 2000
    data_stream = np.random.randint(2, size=size)
    for i in range(999, size):
        data_stream[i] = np.random.randint(8)

    for i in range(size):
        adwin.add_element(data_stream[i])
        if adwin.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

if __name__ == '__main__':
    demo()

