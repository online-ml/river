__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from skmultiflow.filtering.base_filters import MissingValuesCleaner


def demo():
    opt = FileOption('FILE', 'OPT_NAME', '../datasets/covtype.csv', 'csv', False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()

    filter = MissingValuesCleaner(-47, 'median', 10)

    X, y = stream.next_instance(10)

    X[9, 0] = -47

    for i in range(10):
        X[i] = filter.partial_fit_transform([X[i]])

    print(X)

    #for i in range(10):
        #X[i] = filter.partial_fit_transform(X[i])
        #print(X[i])


if __name__ == '__main__':
    demo()
