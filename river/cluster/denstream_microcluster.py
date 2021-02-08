import math
from abc import ABCMeta


class MicroCluster(metaclass=ABCMeta):
    """Micro cluster

    A micro cluster at time `t` for a group of closed points `p_{i_1}, p_{i_2}, ..., p_{i_n}}
    with time stamps `T_{i_1},  T_{i_2}, ..., T_{i_n}` is defined as ${\bar{CF^1}, \bar{CF^2}, \omega}$ [1]_.
        * $\omega = \sum_{j=1}^n f(t - T_{i_j})$ is the weight;
        * $\beta$ is the parameter to determine the threshold of outlier relative to micro clusters;
        * $\bar{CF^1} = \sum_{j=1}^n f(t - T_{i_j}) p_{i_j}$ is the weighted sum of the points;
        * $\bar{CF^2} = \sum_{j=1}^n f(t - T_{i_j}) p^2_{i_j}$ is the weighted squared sum of the points.

    Besides, in order to serve the purpose of calculating the metrics, we also save two other information: the (unweighted)
    linear sum (LS) and the (unweighted) squared sum (SS)
        * $LS = \sum_{j = 1}^n p_{i_j}$
        * $SS = \sum_{j = 1}^n p^2_{i_j}$

    The center of the micro cluster is

    $$
    c = \frac{\bar{CF^1}}{\omega},
    $$

    while the radius of the micro cluster is

    $$
    r = \sqrt{\frac{|\bar{CF^2}|}{\omega} - \left( \frac{|\bar{CF^1}|}{\omega} \right)^2}
    $$

    In order to maintain all the information incrementally, we proceed to calculate the parameters
    (weight, weighted linear sum and weighted squared sum) at the initial state, i.e at the time `t = 0`.
    When the `current_time` parameter is passed, all the parameters will be calculated by simply multiplying
    $f(t)$ to the initial values.

    Parameters
    ----------
    x
        The initial point that is added to the micro cluster

    timestamp
        The timestamp that `x` is created and added to the micro cluster

    decaying_factor
        The decaying factor defined by the user

    current_time
        Current time that the micro cluster is called. At the beginning, when the cluster is created,
        `current_time` is equal to `timestamp`, the time stamp of the first point in the cluster.

    Attributes
    ----------
    N
        number of points in the micro cluster

    dim
        The dimension of the points in the cluster, which is equal to the number of attributes in each point.

    creation_time
        The time point that the micro cluster is created.
        It is equal to the timestamp of the first element of the micro cluster.

    References
    ----------
    .. [1] Feng et al (2006, pp 328-339). Density-Based Clustering over an Evolving Data Stream with Noise.
    In Proceedings of the Sixth SIAM International Conference on Data Mining, April 20-22, ,
     April 20–22, 2006, Bethesda, MD, USA.
    """

    def __init__(self, x=None, timestamp=None, decaying_factor=None, current_time=None):

        self.x = x
        self.timestamp = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor
        self.current_time = current_time

        if x is not None and timestamp is not None:

            "initial weight, initial weighted linear sum (IWLS) and initial weighted squared sum (IWSS) \
            are calculated as described"

            self.N = 1
            self.dim = len(self.x)
            self.initial_weight = 2 ** (self.decaying_factor * self.timestamp)
            self.LS = x
            self.SS = {i: (x[i] * x[i]) for i in range(self.dim)}
            self.IWLS = {
                i: (2 ** (self.decaying_factor * self.timestamp)) * x[i]
                for i in range(self.dim)
            }
            self.IWSS = {
                i: (2 ** (self.decaying_factor * self.timestamp)) * (x[i] * x[i])
                for i in range(self.dim)
            }

    @property
    def weighted_linear_sum(self):
        weighted_linear_sum = {
            i: (2 ** (-self.current_time * self.decaying_factor)) * self.IWLS[i]
            for i in range(self.dim)
        }
        return weighted_linear_sum

    @property
    def weighted_squared_sum(self):
        weighted_squared_sum = {
            i: (2 ** (-self.current_time * self.decaying_factor)) * self.IWSS[i]
            for i in range(self.dim)
        }
        return weighted_squared_sum

    @property
    def weight(self):
        weight = (
            2 ** (-self.current_time * self.decaying_factor)
        ) * self.initial_weight
        return weight

    @property
    def center(self):
        center = {i: self.weighted_linear_sum[i] / self.weight for i in range(self.dim)}
        return center

    @property
    def radius(self):
        radius = math.sqrt(
            abs(
                self.norm(self.weighted_squared_sum) / self.weight
                - (self.norm(self.weighted_linear_sum) / self.weight) ** 2
            )
        )
        return radius

    def add(self, cluster):
        assert self.dim == cluster.dim
        self.N += cluster.N
        self.initial_weight += cluster.initial_weight
        for i in range(self.dim):
            self.LS[i] += cluster.LS[i]
            self.SS[i] += cluster.SS[i]
            self.IWLS[i] += cluster.IWLS[i]
            self.IWSS[i] += cluster.IWSS[i]
        if self.creation_time > cluster.creation_time:
            self.creation_time = cluster.creation_time

    @staticmethod
    def norm(x):
        norm = 0
        for i in range(len(x)):
            norm += x[i] * x[i]
        return math.sqrt(norm)
