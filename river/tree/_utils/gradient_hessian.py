import copy

from river import stats


class GradHess:
    """ The most basic inner structure of the Streaming Gradient Trees. """

    __slots__ = ["gradient", "hessian"]

    def __init__(self, gradient: float = 0.0, hessian: float = 0.0, *, grad_hess=None):
        if grad_hess:
            self.gradient = grad_hess.gradient
            self.hessian = grad_hess.hessian
        else:
            self.gradient = gradient
            self.hessian = hessian

    def __iadd__(self, other):
        self.gradient += other.gradient
        self.hessian += other.hessian

        return self

    def __isub__(self, other):
        self.gradient -= other.gradient
        self.hessian -= other.hessian

        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other
        return new


class GradHessStats:
    """ Class used to monitor and update the gradient/hessian information in Streaming Gradient
    Trees.

    Represents the aggregated gradient/hessian data in a node (global node statistics), category,
    or numerical feature's discretized bin.
    """

    def __init__(self):
        self.x_m = stats.Mean()
        self.g_var = stats.Var()
        self.h_var = stats.Var()
        self.gh_cov = stats.Cov()

    def get_x(self) -> float:
        """ Get the centroid x data that represents all the observations inside a bin. """
        return self.x_m.get()

    def __iadd__(self, other):
        self.x_m += other.x_m
        self.g_var += other.g_var
        self.h_var += other.h_var
        self.gh_cov += other.gh_cov

        return self

    def __isub__(self, other):
        self.x_m -= other.x_m
        self.g_var -= other.g_var
        self.h_var -= other.h_var
        self.gh_cov -= other.gh_cov

        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other

        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other

        return new

    def update(self, gh: GradHess, x=None, w: float = 1.0):
        # Update x values in the case of numerical features (binning strategy)
        if x is not None:
            self.x_m.update(x, w)

        self.g_var.update(gh.gradient, w)
        self.h_var.update(gh.hessian, w)
        self.gh_cov.update(gh.gradient, gh.hessian, w)

    def mean(self) -> GradHess:
        return GradHess(self.g_var.mean.get(), self.h_var.mean.get())

    def variance(self) -> GradHess:
        return GradHess(self.g_var.get(), self.h_var.get())

    def covariance(self) -> float:
        return self.gh_cov.get()

    @property
    def total_weight(self) -> float:
        return self.g_var.mean.n

    # This method ignores correlations between delta_pred and the gradients/hessians! Considering
    # delta_pred is derived from the gradient and hessian sample, this assumption is definitely
    # violated. However, as empirically demonstrated in the original SGT, this fact does not seem
    # to significantly impact on the obtained results.
    def delta_loss_mean_var(self, delta_pred: float) -> stats.Var:
        m = self.mean()
        dlms = stats.Var()
        dlms.mean.n = self.total_weight
        dlms.mean.mean = (
            delta_pred * m.gradient + 0.5 * m.hessian * delta_pred * delta_pred
        )

        variance = self.variance()
        covariance = self.covariance()

        grad_term_var = delta_pred * delta_pred * variance.gradient
        hess_term_var = 0.25 * variance.hessian * (delta_pred ** 4.0)
        dlms.sigma = max(
            0.0, grad_term_var + hess_term_var + (delta_pred ** 3) * covariance
        )
        return dlms
