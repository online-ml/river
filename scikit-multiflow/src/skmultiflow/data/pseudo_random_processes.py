import numpy as np


def random_index_based_on_weights(weights, random_state):
    """ random_index_based_on_weights

    Generates a random index, based on index weights and a random
    generator instance.

    Parameters
    ----------
    weights: list
        The weights of the centroid's indexes.

    random_state: numpy.random
        A random generator.

    Returns
    -------
    int
        The generated index.

    """
    prob_sum = np.sum(weights)
    val = random_state.rand() * prob_sum
    index = 0
    sum_value = 0.0
    while (sum_value <= val) & (index < len(weights)):
        sum_value += weights[index]
        index += 1
    return index - 1
