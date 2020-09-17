from skmultiflow.rules.base_predicate import Predicate
from .instance_conditional_test import InstanceConditionalTest


class NumericAttributeBinaryTest(InstanceConditionalTest):
    def __init__(self, att_idx, att_value, equal_passes_test):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value
        self._equals_passes_test = equal_passes_test

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        v = X[self._att_idx]
        if v == self._att_value:
            return 0 if self._equals_passes_test else 1
        return 0 if v < self._att_value else 1

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch):
        if branch == 0 or branch == 1:
            compare_char = '<' if branch == 0 else '>'
            equals_branch = 0 if self._equals_passes_test else 1
            compare_char += '=' if branch == equals_branch else ''
            return 'Attribute {} {} {}'.format(
                self._att_idx, compare_char, self._att_value
            )

    def branch_rule(self, branch):
        condition = '<' if branch == 0 else '>'
        equals_branch = 0 if self._equals_passes_test else 1
        condition += '=' if branch == equals_branch else ''
        return Predicate(self._att_idx, condition, self._att_value)

    def get_atts_test_depends_on(self):
        return [self._att_idx]

    def get_split_value(self):
        return self._att_value
