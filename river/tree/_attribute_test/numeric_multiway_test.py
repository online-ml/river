import math

from .instance_conditional_test import InstanceConditionalTest


class NumericMultiwayTest(InstanceConditionalTest):
    """Multi-way split tests for numerical features.

    The resulting test considers one branch for each discretized feature bin.

    Parameters
    ----------
    att_idx
        The id of the attribute.
    branch_mapping
        A dictionary that maps the feature values to their respective
        branch ids.
    """

    def __init__(self, att_idx, radius, branch_mapping):
        super().__init__()
        self._att_idx = att_idx
        self._radius = radius
        self._branch_mapping = branch_mapping
        self._reverse_branch_mapping = {b: v for v, b in branch_mapping.items()}

    def branch_for_instance(self, x):
        # Return branch for feature value or -1 in case the element was not
        # observed yet
        return self._branch_mapping.get(math.floor(x[self._att_idx] / self._radius), -1)

    @staticmethod
    def max_branches():
        return -1

    def describe_condition_for_branch(self, branch, shorten=False):
        lower_bound = self._reverse_branch_mapping[branch] * self._radius
        upper_bound = lower_bound + self._radius
        if shorten:
            return f"[{lower_bound}, {upper_bound})"
        else:
            return f"{lower_bound} <= {self._att_idx} < {upper_bound}"

    def attrs_test_depends_on(self):
        return [self._att_idx]

    def add_new_branch(self, att_val):
        """Add a previously unseen feature bin to the test options.

        New branches are created when the projected input value lies outside the range of
        the already discretized intervals.

        Parameters
        ----------
        att_val
            The input value.

        Returns
        -------
            The branch id corresponding for the new value.
        """
        new_branch_id = max(self._branch_mapping.values()) + 1
        # The discretized value
        projection = math.floor(att_val / self._radius)
        self._branch_mapping[projection] = new_branch_id
        self._reverse_branch_mapping[new_branch_id] = projection
        return new_branch_id
