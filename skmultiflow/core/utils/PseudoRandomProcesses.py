import numpy as np

def randomIndexBasedOnWeights(weights, rand):
    probSum = np.sum(weights)
    val = rand.rand() * probSum
    index = 0
    sum = 0.0
    while ((sum <= val) & (index < len(weights))):
        sum += weights[index]
        index += 1
    return index - 1