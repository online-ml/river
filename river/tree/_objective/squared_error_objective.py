from .base_objective import BaseObjective
from .._utils import GradHess


class SquaredErrorObjective(BaseObjective):
    """ Loss function used in regression tasks. """
    def compute_derivatives(self, y, y_pred):
        return GradHess(y_pred - y, 1.0)
