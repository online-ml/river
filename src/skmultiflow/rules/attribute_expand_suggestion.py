class AttributeExpandSuggestion(object):
    def __init__(self, att_idx, att_val, operator, resulting_class_distributions, merit):
        self.resulting_class_distributions = resulting_class_distributions
        self.merit = merit
        self.att_idx = att_idx
        self.att_val = att_val
        self.operator = operator

    def num_splits(self):
        return len(self.resulting_class_distributions)

    def resulting_stats_from_split(self, split_idx):
        return self.resulting_class_distributions[split_idx]
