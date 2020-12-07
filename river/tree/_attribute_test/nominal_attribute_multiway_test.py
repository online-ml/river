from .instance_conditional_test import InstanceConditionalTest


class NominalAttributeMultiwayTest(InstanceConditionalTest):
    """Multi-way split tests for categorical features.

    The resulting test considers one branch for each possible feature
    value.

    Parameters
    ----------
    att_idx
        The id of the attribute.
    branch_mapping
        A dictionary that maps the feature values to their respective
        branch ids.
    """

    def __init__(self, att_idx, branch_mapping):
        super().__init__()
        self._att_idx = att_idx
        self._branch_mapping = branch_mapping
        self._reverse_branch_mapping = {b: v for v, b in branch_mapping.items()}

    def branch_for_instance(self, x):
        # Return branch for feature value or -1 in case the element was not
        # observed yet
        return self._branch_mapping.get(x[self._att_idx], -1)

    @staticmethod
    def max_branches():
        return -1

    def describe_condition_for_branch(self, branch, shorten=False):
        if shorten:
            return f" = {self._reverse_branch_mapping[branch]}"
        else:
            return f"{self._att_idx} = {self._reverse_branch_mapping[branch]}"

    def attrs_test_depends_on(self):
        return [self._att_idx]

    def add_new_branch(self, att_val):
        """Add a previously unseen categorical attribute value to the test
        options.

        Used to create new branches for unseen categories.

        Parameters
        ----------
        att_val
            The value of the new category.

        Returns
        -------
            The branch id corresponding for the new value.
        """
        new_branch_id = max(self._branch_mapping.values()) + 1
        self._branch_mapping[att_val] = new_branch_id
        self._reverse_branch_mapping[new_branch_id] = att_val
        return new_branch_id
