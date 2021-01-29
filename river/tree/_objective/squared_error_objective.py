from .._utils import GradHess
from .base_objective import BaseObjective


class SquaredErrorObjective(BaseObjective):
    """ Loss function used in regression tasks. """

    def compute_derivatives(self, y, y_pred):
        return GradHess(y_pred - y, 1.0)
