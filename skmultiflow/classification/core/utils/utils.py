__author__ = 'Jacob Montiel'

from skmultiflow.classification.core.attribute_class_observers.attribute_class_observer import AttributeClassObserver

def do_naive_bayes_prediction(X, observed_class_distribution:dict, attribute_observers:list):
    votes = {}
    observed_class_sum = sum(observed_class_distribution.values())
    for class_index, observed_class_val in observed_class_distribution.items():
        votes[class_index] = observed_class_val / observed_class_sum
        for att_idx in range(len(X)):
            obs = attribute_observers[att_idx]
            if obs is not None:
                votes[class_index] *= obs.probability_of_attribute_value_given_class(X[att_idx], class_index)
    return votes