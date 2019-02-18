class TopK:
    """Compute online top k.

    This class store in a dictionnary modalities and frequency of k first value of a given series.
    Topk allow to get modality value which have the higher frequency.

    Args:
        k (int): Number of modalities in the target variable.

    """

    def __init__(self, k):
        self.k = k
        self.top = {}

    def update(self, x):
        if x in self.top:
            self.top[x] += 1
        elif len(self.top) <= self.k:
            self.top[x] = 0
        return self

    def get(self):
        return max(self.top, key=self.top.get)
