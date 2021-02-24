import math
import numbers
import sys
from typing import Dict, Hashable, Optional, Union

from river import stats
from river.base.typing import FeatureName, Target

from .._utils import FeatureQuantizer, GradHess, GradHessStats


class SGTSplit:
    """Split Class of the Stochastic Gradient Trees (SGT).

    SGTs only have one type of node. Their nodes, however, can carry a split object that indicates
    they have been split and are not leaves anymore.

    Parameters
    ----------
    feature_idx
        The identifier of the split feature.
    feature_val
        The value of the split.
    is_nominal
        Whether the split feature is nominal.
    """

    def __init__(
        self,
        feature_idx: FeatureName = None,
        feature_val: Target = None,
        is_nominal=False,
    ):
        # loss_mean and loss_var are actually statistics that approximate the *change* in loss.
        self.loss_mean = 0.0
        self.loss_var = 0.0
        self.delta_pred = None

        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.is_nominal = is_nominal


class SGTLearningNode:
    """Learning Node of the Stochastic Gradient Trees (SGT).

    There is only one type of learning node in SGTs. It handles only gradient and hessian
    information about the target. The tree handles target transformation and encoding.

    Parameters
    ----------
    prediction
        The initial prediction of the node.
    depth
        The depth of the node in the tree.

    """

    def __init__(self, prediction=0.0, depth=0):
        self._prediction = prediction
        self.depth = depth
        self.last_split_attempt_at = 0

        # Split test
        self._split: Optional[SGTSplit] = None
        self._children: Optional[Dict[Hashable, "SGTLearningNode"]] = None
        self._split_stats: Optional[
            Dict[FeatureName, Union[Dict[Hashable, GradHessStats], FeatureQuantizer]]
        ] = {}
        self._update_stats = GradHessStats()

    def reset(self):
        self._split_stats = {}
        self._update_stats = GradHessStats()

    def sort_instance_to_leaf(self, x) -> "SGTLearningNode":
        if self._split is None:
            return self

        if self._split.is_nominal:
            try:
                node = self._children[x[self._split.feature_idx]]
            except KeyError:  # Nominal value was not observed previously
                # Create new node to encompass the emerging category
                self._children[x[self._split.feature_idx]] = SGTLearningNode(
                    prediction=self._prediction, depth=self.depth + 1
                )
                node = self._children[x[self._split.feature_idx]]
        else:
            try:
                node = (
                    self._children[0]
                    if x[self._split.feature_idx] <= self._split.feature_val
                    else self._children[1]
                )
            except KeyError:  # Numerical split feature is missing from instance
                # Select the most traversed branch
                branch = max(
                    self._children, key=lambda k: self._children[k].total_weight
                )
                node = self._children[branch]

        return node.sort_instance_to_leaf(x)

    def update(self, x: dict, gh: GradHess, sgt, w: float = 1.0):
        for idx, x_val in x.items():
            if not isinstance(x_val, numbers.Number) or idx in sgt.nominal_attributes:
                # Update the set of nominal features
                sgt.nominal_attributes.add(idx)
                try:
                    self._split_stats[idx][x_val].update(gh, w=w)
                except KeyError:
                    if idx not in self._split_stats:
                        # Categorical features are treated with a simple dict structure
                        self._split_stats[idx] = {}
                    self._split_stats[idx][x_val] = GradHessStats()
                    self._split_stats[idx][x_val].update(gh, w=w)
            else:
                try:
                    self._split_stats[idx].update(x_val, gh, w)
                except KeyError:
                    # Create a new quantizer
                    quantization_radius = sgt._get_quantization_radius(idx)
                    self._split_stats[idx] = FeatureQuantizer(
                        radius=quantization_radius
                    )
                    self._split_stats[idx].update(x_val, gh, w)

        self._update_stats.update(gh, w=w)

    def leaf_prediction(self) -> float:
        return self._prediction

    def find_best_split(self, sgt) -> SGTSplit:
        best_split = SGTSplit()

        # Null split: update the prediction using the new gradient information
        best_split.delta_pred = self.delta_prediction(
            self._update_stats.mean, sgt.lambda_value
        )
        dlms = self._update_stats.delta_loss_mean_var(best_split.delta_pred)
        best_split.loss_mean = dlms.mean.get()
        best_split.loss_var = dlms.get()

        for feature_idx in self._split_stats:
            candidate = SGTSplit()
            candidate.feature_idx = feature_idx

            if feature_idx in sgt.nominal_attributes:
                # Nominal attribute has been already used in a previous split
                if feature_idx in sgt._split_features:
                    continue

                candidate.is_nominal = True
                candidate.delta_pred = {}
                all_dlms = stats.Var()

                cat_collection = self._split_stats[feature_idx]
                for category in cat_collection:
                    dp = self.delta_prediction(
                        cat_collection[category].mean, sgt.lambda_value
                    )

                    dlms = cat_collection[category].delta_loss_mean_var(dp)
                    candidate.delta_pred[category] = dp

                    all_dlms += dlms

                candidate.loss_mean = (
                    all_dlms.mean.get()
                    + len(cat_collection) * sgt.gamma / self.total_weight
                )
                candidate.loss_var = all_dlms.get()
            else:  # Numerical features
                quantizer = self._split_stats[feature_idx]
                half_radius = quantizer.radius / 2.0
                n_bins = len(quantizer)
                if n_bins == 1:  # Insufficient number of bins to perform splits
                    continue

                candidate.loss_mean = math.inf
                candidate.delta_pred = {}

                # Auxiliary gradient and hessian statistics
                left_ghs = GradHessStats()
                left_dlms = stats.Var()
                for i, ghs in enumerate(quantizer):
                    left_ghs += ghs
                    left_delta_pred = self.delta_prediction(left_ghs.mean, sgt.lambda_value)
                    left_dlms += left_ghs.delta_loss_mean_var(left_delta_pred)

                    right_ghs = self._update_stats - left_ghs
                    right_delta_pred = self.delta_prediction(
                        right_ghs.mean, sgt.lambda_value
                    )
                    right_dlms = right_ghs.delta_loss_mean_var(right_delta_pred)

                    all_dlms = left_dlms + right_dlms

                    loss_mean = all_dlms.mean.get()
                    loss_var = all_dlms.get()

                    if loss_mean < candidate.loss_mean:
                        candidate.loss_mean = (
                            loss_mean + 2.0 * sgt.gamma / self.total_weight
                        )
                        candidate.loss_var = loss_var
                        candidate.delta_pred[0] = left_delta_pred
                        candidate.delta_pred[1] = right_delta_pred

                        # Define split point
                        if i == n_bins - 1:  # Last bin
                            candidate.feature_val = ghs.centroid_x
                        else:  # Use middle point between bins
                            candidate.feature_val = ghs.centroid_x + half_radius

            if candidate.loss_mean < best_split.loss_mean:
                best_split = candidate

        return best_split

    def apply_split(self, split, sgt):
        # Null split: update tree prediction and reset learning node
        if split.feature_idx is None:
            self._prediction += split.delta_pred
            sgt._n_node_updates += 1
            self.reset()
            return

        self._split = split
        sgt._n_splits += 1
        sgt._split_features.add(split.feature_idx)

        # Create children
        self._children = {}
        for child_idx, delta_pred in split.delta_pred.items():
            self._children[child_idx] = SGTLearningNode(
                prediction=self._prediction + delta_pred, depth=self.depth + 1
            )
        # Free memory used to monitor splits
        self._split_stats = None

        # Update max depth
        if self.depth + 1 > sgt._depth:
            sgt._depth = self.depth + 1

    def is_leaf(self) -> bool:
        return self._children is None

    @property
    def children(self) -> Dict[FeatureName, "SGTLearningNode"]:
        return self._children

    @property
    def total_weight(self) -> float:
        return self._update_stats.total_weight
    
    @property
    def update_stats(self):
        return self._update_stats

    @staticmethod
    def delta_prediction(gh: GradHess, lambda_value: float):
        # Add small constant value to avoid division by zero
        return -gh.gradient / (gh.hessian + sys.float_info.min + lambda_value)
