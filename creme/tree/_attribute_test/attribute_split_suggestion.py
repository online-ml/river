class AttributeSplitSuggestion(object):
    def __init__(self, split_test, resulting_class_distributions, merit):
        self.split_test = split_test
        self.resulting_class_distributions = resulting_class_distributions
        self.merit = merit

    def num_splits(self):
        return len(self.resulting_class_distributions)

    def resulting_stats_from_split(self, split_idx):
        return self.resulting_class_distributions[split_idx]
