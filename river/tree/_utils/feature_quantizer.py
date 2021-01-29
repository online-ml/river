import math

from .gradient_hessian import GradHess, GradHessStats


class FeatureQuantizer:
    """ Adapted version of the Quantizer Observer (QO) that is applied to SGTs [^1].

    Parameters
    ----------
    radius
        The quantization radius.

    References
    ----------
    [^1]: Mastelini, S.M. and de Carvalho, A.C.P.D.L.F., 2020. Using dynamical quantization to
    perform split attempts in online tree regressors. arXiv preprint arXiv:2012.00083.
    """
    def __init__(self, radius):
        self.radius = radius
        self.hash = {}

    def __getitem__(self, k):
        return self.hash[k]

    def __len__(self):
        return len(self.hash)

    def update(self, x_val, gh: GradHess, w: float):
        index = math.floor(x_val / self.radius)
        if index in self.hash:
            self.hash[index].update(gh=gh, x=x_val, w=w)
        else:
            ghs = GradHessStats()
            ghs.update(gh=gh, x=x_val, w=w)
            self.hash[index] = ghs

    def __iter__(self):
        for k in sorted(self.hash):
            yield self.hash[k]
