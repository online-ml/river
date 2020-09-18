import math


def do_naive_bayes_prediction(X, observed_class_distribution: dict, attribute_observers: dict):
    """
    Perform Naive Bayes prediction

    Parameters
    ----------
    X: numpy.ndarray, shape (1, n_features)
        A feature's vector.

    observed_class_distribution: dict
        Observed class distribution

    attribute_observers: dict
        Attribute (features) observer

    Returns
    -------
    votes
        dict

    Notes
    -----
    This method is not intended to be used as a stand-alone method.
    """
    observed_class_sum = sum(observed_class_distribution.values())
    if observed_class_distribution == {} or observed_class_sum == 0.0:
        # No observed class distributions, all classes equal
        return {0: 0.0}
    votes = {}
    for class_index, observed_class_val in observed_class_distribution.items():
        votes[class_index] = observed_class_val / observed_class_sum
        if attribute_observers:
            for att_idx in range(len(X)):
                if att_idx in attribute_observers:
                    obs = attribute_observers[att_idx]
                    tmp = votes[class_index] * obs.probability_of_attribute_value_given_class(
                        X[att_idx], class_index)
                    votes[class_index] = tmp if not math.isnan(tmp) else 0
    return votes
