from . import base
from . import quantile


class IQR(base.Univariate):
    def __init__(self, q_inf=0.25, q_sup=0.75):
        if q_inf >= q_sup:
            raise ValueError('q_inf must be strictly less than q_sup')
        self.quantile_inf = quantile.Quantile(quantile=q_inf)
        self.quantile_sup = quantile.Quantile(quantile=q_sup)

    def update(self, x):
        self.quantile_inf.update(x)
        self.quantile_sup.update(x)
        return self

    def get(self):
        return self.quantile_sup.get() - self.quantile_inf.get()
