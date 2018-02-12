__author__ = 'Jacob Montiel'

import math

def do_naive_bayes_prediction(X, observed_class_distribution: dict, attribute_observers: list):
    if observed_class_distribution == {}:
        # No observed class distributions, all classes equal
        return {0: 0.0}
    votes = {}
    observed_class_sum = sum(observed_class_distribution.values())
    for class_index, observed_class_val in observed_class_distribution.items():
        votes[class_index] = observed_class_val / observed_class_sum
        if attribute_observers != []:
            for att_idx in range(len(X)):
                obs = attribute_observers[att_idx]
                if obs is not None:
                    votes[class_index] *= obs.probability_of_attribute_value_given_class(X[att_idx], class_index)
    return votes


def normalize_values_in_dict(sum_value, dictionary):
    if sum == 0:
        raise ValueError('Can not normalize array. Sum is zero')
    if math.isnan(sum_value):
        raise ValueError('Can not normalize array. Sum is NaN')
    for key, value in dictionary.items():  # loop over the keys, values in the dictionary
        dictionary[key] = value / sum_value


#Return the max index of max value
def get_max_value_index(dictionary):
    maximum = 0
    max_idx = 0
    for key, value in dictionary.items():
        if value > maximum:
            maximum = value
            max_idx = key
    return max_idx
