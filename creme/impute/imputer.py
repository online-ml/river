from abc import ABC

__all__ = ['Imputer']


class Imputer(ABC):
    """ Abstract class to impute missing values

    Args:
        None

    """

    def __init__(self):
        pass

    def fit_one(self, x):
        if self.on in x:
            key = x[self.by] if self.by else None
            self.imputers[key].update(x[self.on])
            return x
        return self.transform_one(x)

    def transform_one(self, x):
        if self.on not in x:
            key = x[self.by] if self.by else None
            return {
                **x,
                self.on: self.imputers[key].get()
            }
        return x
