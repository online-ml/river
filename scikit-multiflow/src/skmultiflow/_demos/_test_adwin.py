import numpy as np
from river.drift import ADWIN


def demo():
    """ _test_adwin

    In this demo, an ADWIN object evaluates a sequence of numbers corresponding to 2 distributions.
    The ADWIN object indicates the indices where change is detected.

    The first half of the data is a sequence of randomly generated 0's and 1's.
    The second half of the data is a normal distribution of integers from 0 to 7.

    """
    adwin = ADWIN()
    size = 2000
    change_start = 999
    np.random.seed(1)
    data_stream = np.random.randint(2, size=size)
    data_stream[change_start:] = np.random.randint(8, size=size-change_start)

    for i in range(size):
        change_detected, _ = adwin.update(data_stream[i])
        if change_detected:
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

if __name__ == '__main__':
    demo()
