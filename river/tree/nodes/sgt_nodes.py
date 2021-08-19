import collections
import math
import numbers
import sys
from typing import Dict, Hashable, Optional, Union

from river import stats
from river.base.typing import FeatureName

from ..base import Leaf
from ..splitter.sgt_quantizer import DynamicQuantizer, StaticQuantizer
from ..utils import BranchFactory, GradHess, GradHessMerit, GradHessStats
from .branch import NominalMultiwayBranch, NumericBinaryBranch


class SGTLeaf(Leaf):
    """Leaf Node of the Stochastic Gradient Trees (SGT).

    There is only one type of leaf in SGTs. It handles only gradient and hessian
    information about the target. The tree handles target transformation and encoding.

    Parameters
    ----------
    prediction
        The initial prediction of the leaf.
    depth
        The depth of the leaf in the tree.
    split_params
        Parameters passed to the feature quantizers.
    """

    def __init__(self, prediction=0.0, depth=0, split_params=None):
        super().__init__()
        self._prediction = prediction
        self.depth = depth

        # Quantizer params are dynamically updated
        self.split_params = (
            split_params if split_params is not None else collections.defaultdict(dict)
        )
        self.last_split_attempt_at = 0

        self._split_stats: Optional[
            Dict[
                FeatureName,
                Union[Dict[Hashable, GradHessStats], DynamicQuantizer, StaticQuantizer],
            ]
        ] = {}
        self._update_stats = GradHessStats()

    def reset(self):
        self._split_stats = {}
        self._update_stats = GradHessStats()

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

                    # TODO check here
                    # Create a new quantizer
                    quantization_radius = sgt._get_quantization_radius(idx)
                    self._split_stats[idx] = DynamicQuantizer(
                        radius=quantization_radius
                    )
                    self._split_stats[idx].update(x_val, gh, w)

        self._update_stats.update(gh, w=w)

    def prediction(self) -> float:
        return self._prediction

    def find_best_split(self, sgt) -> BranchFactory:
        best_split = BranchFactory()
        best_split.merit = GradHessMerit()

        # Null split: update the prediction using the new gradient information
        best_split.merit.delta_pred = self.delta_prediction(
            self._update_stats.mean, sgt.lambda_value
        )
        dlms = self._update_stats.delta_loss_mean_var(best_split.merit.delta_pred)
        best_split.merit.loss_mean = dlms.mean.get()
        best_split.merit.loss_var = dlms.get()

        for feature_idx in self._split_stats:
            candidate = BranchFactory()
            candidate.merit = GradHessMerit()
            candidate.feature = feature_idx

            if feature_idx in sgt.nominal_attributes:
                # Nominal attribute has been already used in a previous split
                if feature_idx in sgt._split_features:
                    continue

                candidate.numerical_feature = False
                candidate.merit.delta_pred = {}
                all_dlms = stats.Var()

                cat_collection = self._split_stats[feature_idx]
                candidate.split_info = list(cat_collection.keys())
                for category in cat_collection:
                    dp = self.delta_prediction(
                        cat_collection[category].mean, sgt.lambda_value
                    )

                    dlms = cat_collection[category].delta_loss_mean_var(dp)
                    candidate.merit.delta_pred[category] = dp

                    all_dlms += dlms

                candidate.merit.loss_mean = (
                    all_dlms.mean.get()
                    + len(cat_collection) * sgt.gamma / self.total_weight
                )
                candidate.merit.loss_var = all_dlms.get()
            else:  # Numerical features
                quantizer = self._split_stats[feature_idx]

                # Get updated quantizer params
                self.split_params[feature_idx].update(quantizer._get_params())

                # TODO check here
                half_radius = quantizer.radius / 2.0
                n_bins = len(quantizer)
                if n_bins == 1:  # Insufficient number of bins to perform splits
                    continue

                candidate.merit.loss_mean = math.inf
                candidate.merit.delta_pred = {}

                # Auxiliary gradient and hessian statistics
                left_ghs = GradHessStats()
                left_dlms = stats.Var()
                for i, ghs in enumerate(quantizer):
                    left_ghs += ghs
                    left_delta_pred = self.delta_prediction(
                        left_ghs.mean, sgt.lambda_value
                    )
                    left_dlms += left_ghs.delta_loss_mean_var(left_delta_pred)

                    right_ghs = self._update_stats - left_ghs
                    right_delta_pred = self.delta_prediction(
                        right_ghs.mean, sgt.lambda_value
                    )
                    right_dlms = right_ghs.delta_loss_mean_var(right_delta_pred)

                    all_dlms = left_dlms + right_dlms

                    loss_mean = all_dlms.mean.get()
                    loss_var = all_dlms.get()

                    if loss_mean < candidate.merit.loss_mean:
                        candidate.merit.loss_mean = (
                            loss_mean + 2.0 * sgt.gamma / self.total_weight
                        )
                        candidate.merit.loss_var = loss_var
                        candidate.merit.delta_pred[0] = left_delta_pred
                        candidate.merit.delta_pred[1] = right_delta_pred

                        # TODO check that too

                        # Define split point
                        if i == n_bins - 1:  # Last bin
                            candidate.split_info = ghs.centroid_x
                        else:  # Use middle point between bins
                            candidate.split_info = ghs.centroid_x + half_radius

            if candidate.merit.loss_mean < best_split.merit.loss_mean:
                best_split = candidate

        return best_split

    def apply_split(self, split, p_node, p_branch, sgt):
        # Null split: update tree prediction and reset learning node
        if split.feature is None:
            self._prediction += split.delta_pred
            sgt._n_node_updates += 1
            self.reset()
            return

        sgt._n_splits += 1
        sgt._split_features.add(split.feature_idx)

        branch = (
            NumericBinaryBranch if split.numerical_feature else NominalMultiwayBranch
        )
        child_depth = self.depth + 1
        leaves = tuple(
            SGTLeaf(self._prediction + delta_pred, depth=child_depth)
            for delta_pred in split.merit.delta_pred.values()
        )

        new_split = split.assemble(branch, self.split_params, self.depth, *leaves)

        if p_branch is None:
            sgt._root = new_split
        else:
            p_node.children[p_branch] = new_split

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
