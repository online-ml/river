__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.classification.core.driftdetection.adwin import ADWIN


def demo():
    """ _test_adwin
    
    This demo will insert data into an ADWIN object when will display in which 
    indexes change was detected.
    
    The data stream is simulated as a sequence of randomly generated 0's and 1's. 
    Then the data from indexes 999 to 1999 is changed to a normal distribution of 
    integers from 0 to 7.
    
    """
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

