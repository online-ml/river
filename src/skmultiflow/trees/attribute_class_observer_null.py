from skmultiflow.trees.attribute_class_observer import AttributeClassObserver


class AttributeClassObserverNull(AttributeClassObserver):
    """ Class for observing the class data distribution for a null attribute.
    This method is used to disable the observation for an attribute.
    Used in decision trees to monitor data statistics on leaves.

    """
    def __init__(self):
        super().__init__()

    def observe_attribute_class(self, att_val, class_val, weight):
        pass

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        return None
