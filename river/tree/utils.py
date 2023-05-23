from __future__ import annotations

import collections
import copy
import dataclasses
import functools
import math
import sys
import typing

from river.base.typing import FeatureName
from river.stats import Cov, Var


def do_naive_bayes_prediction(x, observed_class_distribution: dict, splitters: dict):
    """Perform Naive Bayes prediction

    Parameters
    ----------
    x
        The feature values.

    observed_class_distribution
        Observed class distribution.

    splitters
        Attribute (features) observers.

    Returns
    -------
    The probabilities related to each class.

    Notes
    -----
    This method is not intended to be used as a stand-alone method.
    """
    total_weight = sum(observed_class_distribution.values())
    if not observed_class_distribution or total_weight == 0:
        # No observed class distributions, all classes equal
        return {}

    votes = {}
    for class_index, class_weight in observed_class_distribution.items():
        # Prior
        if class_weight > 0:
            votes[class_index] = math.log(class_weight / total_weight)
        else:
            votes[class_index] = 0.0
            continue

        if splitters:
            for att_idx in splitters:
                if att_idx not in x:
                    continue
                obs = splitters[att_idx]
                # Prior plus the log likelihood
                tmp = obs.cond_proba(x[att_idx], class_index)
                votes[class_index] += math.log(tmp) if tmp > 0 else 0.0

    # Max log-likelihood
    max_ll = max(votes.values())
    # Apply the log-sum-exp trick (https://stats.stackexchange.com/a/253319)
    lse = max_ll + math.log(sum(math.exp(log_proba - max_ll) for log_proba in votes.values()))

    for class_index in votes:
        votes[class_index] = math.exp(votes[class_index] - lse)

    return votes


@functools.total_ordering
@dataclasses.dataclass
class BranchFactory:
    """Helper class used to assemble branches designed by the splitters.

    If constructed using the default values, a null-split suggestion is assumed.
    """

    merit: float = -math.inf
    feature: FeatureName | None = None
    split_info: typing.Hashable | list[typing.Hashable] | tuple[
        typing.Hashable, list[typing.Hashable]
    ] | None = None
    children_stats: list | None = None
    numerical_feature: bool = True
    multiway_split: bool = False

    def assemble(
        self,
        branch,  # typing.Type[DTBranch],
        stats: dict | Var,
        depth: int,
        *children,
        **kwargs,
    ):
        return branch(stats, self.feature, self.split_info, depth, *children, **kwargs)

    def __lt__(self, other):
        return self.merit < other.merit

    def __eq__(self, other):
        return self.merit == other.merit


class GradHess:
    """The most basic inner structure of the Stochastic Gradient Trees that carries information
    about the gradient and hessian of a given observation.
    """

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


@functools.total_ordering
@dataclasses.dataclass
class GradHessMerit:
    """Class used to keep the split merit of each split candidate, accordingly to its
    gradient and hessian information.

    In Stochastic Gradient Trees, the split merit is given by a combination of the loss mean and
    variance. Additionally, the loss in each resulting tree branch is also accounted.
    """

    loss_mean: float = 0.0
    loss_var: float = 0.0
    delta_pred: float | dict | None = None

    def __lt__(self, other):
        return self.loss_mean < other.loss_mean

    def __eq__(self, other):
        return self.loss_mean == other.loss_mean


class GradHessStats:
    """Class used to monitor and update the gradient/hessian information in Stochastic Gradient
    Trees.

    Represents the aggregated gradient/hessian data in a node (global node statistics), category,
    or numerical feature's discretized bin.
    """

    def __init__(self):
        self.g_var = Var()
        self.h_var = Var()
        self.gh_cov = Cov()

    def __iadd__(self, other):
        self.g_var += other.g_var
        self.h_var += other.h_var
        self.gh_cov += other.gh_cov

        return self

    def __isub__(self, other):
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

    def update(self, gh: GradHess, w: float = 1.0):
        self.g_var.update(gh.gradient, w)
        self.h_var.update(gh.hessian, w)
        self.gh_cov.update(gh.gradient, gh.hessian, w)

    @property
    def mean(self) -> GradHess:
        return GradHess(self.g_var.mean.get(), self.h_var.mean.get())

    @property
    def variance(self) -> GradHess:
        return GradHess(self.g_var.get(), self.h_var.get())

    @property
    def covariance(self) -> float:
        return self.gh_cov.get()

    @property
    def total_weight(self) -> float:
        return self.g_var.mean.n

    # This method ignores correlations between delta_pred and the gradients/hessians! Considering
    # delta_pred is derived from the gradient and hessian sample, this assumption is definitely
    # violated. However, as empirically demonstrated in the original SGT, this fact does not seem
    # to significantly impact on the obtained results.
    def delta_loss_mean_var(self, delta_pred: float) -> Var:
        m = self.mean
        n = self.total_weight
        mean = delta_pred * m.gradient + 0.5 * m.hessian * delta_pred * delta_pred

        variance = self.variance
        covariance = self.covariance

        grad_term_var = delta_pred * delta_pred * variance.gradient
        hess_term_var = 0.25 * variance.hessian * (delta_pred**4.0)
        sigma = max(0.0, grad_term_var + hess_term_var + (delta_pred**3) * covariance)
        return Var._from_state(n, mean, sigma)  # noqa


def calculate_object_size(obj: typing.Any, unit: str = "byte") -> int:
    """Iteratively calculates the `obj` size in bytes.

    Visits all the elements related to obj accounting for their respective
    sizes.

    Parameters
    ----------
    obj
        Object to evaluate.
    unit
        The unit in which the accounted value is going to be returned.
        Values: 'byte', 'kB', 'MB' (Default: 'byte').

    Returns
    -------
    The size of the object and its related properties and objects, in 'unit'.

    """
    seen = set()
    to_visit: collections.deque = collections.deque()
    byte_size = 0

    to_visit.append(obj)

    while True:
        try:
            obj = to_visit.popleft()
        except IndexError:
            break

        # If element was already covered, skip it
        if id(obj) in seen:
            continue

        # Update size accounting
        byte_size += sys.getsizeof(obj)

        # Mark element as seen
        seen.add(id(obj))

        # Add keys and values for size account
        if isinstance(obj, dict):
            for v in obj.values():
                to_visit.append(v)

            for k in obj.keys():
                to_visit.append(k)
        elif hasattr(obj, "__dict__"):
            to_visit.append(obj.__dict__)
        elif hasattr(obj, "__iter__") and not (
            isinstance(obj, str) or isinstance(obj, bytes) or isinstance(obj, bytearray)
        ):
            for i in obj:
                to_visit.append(i)

    if unit == "kB":
        final_size = byte_size / 1024
    elif unit == "MB":
        final_size = byte_size / (2**20)
    else:
        final_size = byte_size

    return int(final_size)


def add_dict_values(dict_a: dict, dict_b: dict, inplace=False) -> dict:
    """Add two dictionaries, summing the values of elements with the same key.

    This function iterates over the keys of dict_b and adds their corresponding
    values to the elements in dict_a. If dict_b has a (key, value) pair that
    does not belong to dict_a, this pair is added to the latter dictionary.

    Parameters
    ----------
    dict_a
        dictionary to update.
    dict_b
        dictionary whose values will be added to `dict_a`.
    inplace
        If `True`, the addition is performed in-place and results are stored in `dict_a`.
        If `False`, `dict_a` is not changed and the results are returned in a new dictionary.

    Returns
    -------
    A dictionary containing the result of the operation. Either a pointer to
    `dict_a` or a new dictionary depending on parameter `inplace`.

    """
    if inplace:
        result = dict_a
    else:
        result = copy.deepcopy(dict_a)

    for k, v in dict_b.items():
        try:
            result[k] += v
        except KeyError:
            result[k] = v
    return result


def round_sig_fig(x, significant_digits=2) -> float:
    """Round considering of significant figures of x, given the select
    `significant_digits` prototype.

    If`significant_digits` match the number of significant figures in `x`, its value
    will be used for rounding; otherwise, decimal places will removed
    accordingly to the significant figures in `x`.

    Parameters
    ----------
    x
        A floating point scalar.
    significant_digits
        The number of intended rounding figures.

    Returns
    -------
        The rounded value of `x`.

    Examples
    --------
    >>> round_sig_fig(1.2345)
    1.2
    >>> round_sig_fig(1.2345, significant_digits=3)
    1.23
    >>> round_sig_fig(0.0)
    0.0
    >>> round_sig_fig(0)
    0
    >>> round_sig_fig(1999, significant_digits=1)
    2000
    >>> round_sig_fig(1999, significant_digits=4)
    1999
    >>> round_sig_fig(0.025, significant_digits=3)
    0.03
    >>> round_sig_fig(0.025, significant_digits=10)
    0.025
    >>> round_sig_fig(0.0250, significant_digits=10)
    0.025
    """
    return round(x, significant_digits - int(math.floor(math.log10(abs(x) + 1))) - 1)
