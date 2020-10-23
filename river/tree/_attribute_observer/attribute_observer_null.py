from .attribute_observer import AttributeObserver


class AttributeObserverNull(AttributeObserver):
    """Dummy observer for attributes that were disabled in the trees."""
    def __init__(self):
        super().__init__()

    def update(self, att_val, target_val, sample_weight):
        pass

    def probability_of_attribute_value_given_class(self, att_val, target_val):
        return 0.0

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        return None
