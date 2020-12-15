from river.utils.skmultiflow_utils import round_sig_fig

from .instance_conditional_test import InstanceConditionalTest


class NumericAttributeBinaryTest(InstanceConditionalTest):
    def __init__(self, att_idx, att_value, equal_passes_test):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value
        self._equals_passes_test = equal_passes_test

    def branch_for_instance(self, x):
        if self._att_idx not in x:
            return -1
        v = x[self._att_idx]
        if v == self._att_value:
            return 0 if self._equals_passes_test else 1
        return 0 if v < self._att_value else 1

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch, shorten=False):
        if branch == 0 or branch == 1:
            compare_char = "<" if branch == 0 else ">"
            equals_branch = 0 if self._equals_passes_test else 1
            compare_char += "=" if branch == equals_branch else ""

            if shorten:
                return f"{compare_char} {round_sig_fig(self._att_value)}"
            else:
                return f"{self._att_idx} {compare_char} {self._att_value}"

    def attrs_test_depends_on(self):
        return [self._att_idx]

    @property
    def split_value(self):
        return self._att_value
