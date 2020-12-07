import math

from river.utils.skmultiflow_utils import normalize_values_in_dict


def do_naive_bayes_prediction(x, observed_class_distribution: dict, attribute_observers: dict):
    """Perform Naive Bayes prediction

    Parameters
    ----------
    x
        The feature values.

    observed_class_distribution
        Observed class distribution

    attribute_observers
        Attribute (features) observer

    Returns
    -------
    votes
        dict

    Notes
    -----
    This method is not intended to be used as a stand-alone method.
    """
    total_weight_sum = sum(observed_class_distribution.values())
    if not observed_class_distribution or total_weight_sum == 0:
        # No observed class distributions, all classes equal
        return None
    votes = {}
    for class_index, class_weight_sum in observed_class_distribution.items():
        # Prior
        votes[class_index] = (
            math.log(class_weight_sum / total_weight_sum) if class_weight_sum > 0 else 0.0
        )
        if attribute_observers:
            for att_idx in attribute_observers:
                if att_idx not in x:
                    continue
                obs = attribute_observers[att_idx]
                # Prior plus the log likelihood
                tmp = obs.probability_of_attribute_value_given_class(x[att_idx], class_index)
                votes[class_index] += math.log(tmp) if tmp > 0 else 0.0
        # Revert log likelihood
        votes[class_index] = math.exp(votes[class_index])
    return normalize_values_in_dict(votes)
