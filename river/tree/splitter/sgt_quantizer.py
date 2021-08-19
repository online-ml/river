import copy
import math
import typing

from river import stats

from .base import Quantizer
from ..utils import GradHess, GradHessStats


class FeatureQuantizer(Quantizer):
    """ Adapted version of the Quantizer Observer (QO) that is applied to SGTs [^1].

    Parameters
    ----------
    radius
        The quantization radius.
    std_prop
        The proportion of the standard deviation that is going to be used to define the radius
        value.


    References
    ----------
    [^1]: Mastelini, S.M. and de Carvalho, A.C.P.D.L.F., 2020. Using dynamical quantization to
    perform split attempts in online tree regressors. arXiv preprint arXiv:2012.00083.
    """

    def __init__(self, radius: float=0.5, std_prop: float=0.25):
        super().__init__()
        self.radius = radius
        self.std_prop = std_prop

        self.feat_var = stats.Var()
        self.hash = {}

    def __getitem__(self, k):
        return self.hash[k]

    def update(self, x_val, gh: GradHess, w: float):
        self.feat_var.update(x_val, w)

        index = math.floor(x_val / self.radius)
        if index in self.hash:
            self.hash[index].update(gh=gh, x=x_val, w=w)
        else:
            # Update the branches
            self.splits.add(index)
            ghs = GradHessStats()
            ghs.update(gh=gh, x=x_val, w=w)
            self.hash[index] = ghs

    def __iter__(self) -> typing.Iterator[GradHessStats]:
        for k in sorted(self.hash):
            yield self.hash[k]

    def __deepcopy__(self, memo):
        new = copy.deepcopy(self)

        std = math.sqrt(self.feat_var.get())

        new_radius = self.std_prop * std
        new.radius = new_radius

        return new





