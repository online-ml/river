from .. import stats
from . import base


class RollingMetric:

    def __init__(self, metric, window_size):
        a = type(metric).__bases__