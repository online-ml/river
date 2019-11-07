from skmultiflow.trees.attribute_test import InstanceConditionalTest
from skmultiflow.rules.base_predicate import Predicate


class NominalAttributeBinaryTest(InstanceConditionalTest):
    def __init__(self, att_idx, att_value):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        else:
            return 0 if int(X[self._att_idx]) == self._att_value else 1

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch):
        condition = ' = ' if branch == 0 else ' != '
        return 'Attribute {}{}{}'.format(self._att_idx, condition, self._att_value)

    def branch_rule(self, branch):
        condition = '==' if branch == 0 else '!='
        return Predicate(self._att_idx, condition, self._att_value)

    def get_atts_test_depends_on(self):
        return [self._att_idx]
