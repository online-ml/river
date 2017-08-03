__author__ = 'Guilherme Matsumoto'

import warnings, logging
import numpy as np
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin
from skmultiflow.data.generators.sea_generator import SEAGenerator


def demo():
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore", ".*Passing 1d.*")
    stream = SEAGenerator(1, noise_percentage=6.7)
    stream.prepare_for_use()
    clf = LeverageBagging(h=KNN(k=8, max_window_size=2000, leaf_size=30), ensemble_length=1)
    sample_count = 0
    correctly_classified = 0
    max_samples = 2000
    train_size = 200
    first = True
    if train_size > 0:
        X, y = stream.next_instance(train_size)
        clf.partial_fit(X, y, classes=stream.get_classes())
        first = False

    logging.info('%s%%', 0.0)
    while sample_count < max_samples:
        if (sample_count+1) % (max_samples / 20) == 0:
            logging.info('%s%%', str(((sample_count // (max_samples / 20)+1) * 5)))
        X, y = stream.next_instance(2)
        my_pred = clf.predict(X)
        if first:
            clf.partial_fit(X, y, classes=stream.get_classes())
            first = False
        else:
            clf.partial_fit(X, y)

        if my_pred is not None:
            if y[0] == my_pred[0]:
                correctly_classified += 1

        sample_count += 1

    print(str(sample_count) + ' samples analyzed.')
    print('My performance: ' + str(correctly_classified / sample_count))

if __name__ == '__main__':
    demo()