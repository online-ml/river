__author__ = 'Guilherme Matsumoto'

import numpy as np


def hamming_score(true_labels, predicts):
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum((true_labels == predicts) * 1.) / N / L

def j_index():
    pass

def exact_match():
    pass
