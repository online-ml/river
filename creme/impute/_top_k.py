class TopK:
    """Compute online top k.

    TopK allow to compute most frequent value in .
    K first value of a given series of value will be stored.
    Increase K allow to increase confidence.

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
