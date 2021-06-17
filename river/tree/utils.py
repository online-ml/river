import dataclasses
import functools
import math
import typing

from river.base.typing import FeatureName
from river.stats import Var


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
        return None

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
    lse = max_ll + math.log(
        sum(math.exp(log_proba - max_ll) for log_proba in votes.values())
    )

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
    feature: typing.Optional[FeatureName] = None
    split_info: typing.Optional[
        typing.Union[
            typing.Hashable,
            typing.List[typing.Hashable],
            typing.Tuple[typing.Hashable, typing.List[typing.Hashable]],
        ]
    ] = None
    children_stats: typing.Optional[typing.List] = None
    numerical_feature: bool = True
    multiway_split: bool = False

    def assemble(
        self,
        branch,  # : typing.Type[HTBranch],
        stats: typing.Union[typing.Dict, Var],
        depth: int,
        *children,
        **kwargs
    ):
        return branch(stats, self.feature, self.split_info, depth, *children, **kwargs)

    def __lt__(self, other):
        return self.merit < other.merit

    def __eq__(self, other):
        return self.merit == other.merit
