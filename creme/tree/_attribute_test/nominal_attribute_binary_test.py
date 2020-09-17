from skmultiflow.rules.base_predicate import Predicate
from .instance_conditional_test import InstanceConditionalTest


class NominalAttributeBinaryTest(InstanceConditionalTest):
    """ Implements binary split tests for categorical features.

        The resulting test considers two branches: one encompassing a specific
        feature value, and another for the remaining cases.

        Parameters
        ----------
            att_idx: int
                The column id for the attribute.
            att_value: float or int
                The categorical value of the feature to test.
    """
    def __init__(self, att_idx, att_value):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        else:
            return 0 if X[self._att_idx] == self._att_value else 1

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch):
        condition = ' = ' if branch == 0 else ' != '
        return 'Attribute {}{}{}'.format(
            self._att_idx, condition, self._att_value
        )

    def branch_rule(self, branch):
        condition = '==' if branch == 0 else '!='
        return Predicate(self._att_idx, condition, self._att_value)

    def get_atts_test_depends_on(self):
        return [self._att_idx]
