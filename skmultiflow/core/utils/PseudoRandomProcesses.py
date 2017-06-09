__author__ = 'Guilherme Matsumoto'

import numpy as np


def random_index_based_on_weights(weights, rand):
    prob_sum = np.sum(weights)
    val = rand.rand() * prob_sum
    index = 0
    sum = 0.0
    while ((sum <= val) & (index < len(weights))):
        sum += weights[index]
        index += 1
    return index - 1