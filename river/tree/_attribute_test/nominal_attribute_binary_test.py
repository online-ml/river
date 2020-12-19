from .instance_conditional_test import InstanceConditionalTest


class NominalAttributeBinaryTest(InstanceConditionalTest):
    """Implement binary split tests for categorical features.

    The resulting test considers two branches: one encompassing a specific
    feature value, and another for the remaining cases.

    Parameters
    ----------
    att_idx
        The id of the attribute.
    att_value
        The categorical value of the feature to test.
    """

    def __init__(self, att_idx, att_value):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value

    def branch_for_instance(self, x):
        try:
            return 0 if x[self._att_idx] == self._att_value else 1
        except KeyError:
            return -1

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch, shorten=False):
        condition = " = " if branch == 0 else " != "
        if shorten:
            return f"{condition}{self._att_value}"
        else:
            return f"{self._att_idx}{condition}{self._att_value}"

    def attrs_test_depends_on(self):
        return [self._att_idx]
