from skmultiflow.trees.instance_conditional_test import InstanceConditionalTest


class NominalAttributeMultiwayTest(InstanceConditionalTest):
    def __init__(self, att_idx):
        super().__init__()
        self._att_idx = att_idx

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        else:
            return X[self._att_idx]

    @staticmethod
    def max_branches():
        return -1

    def describe_condition_for_branch(self, branch):
        return 'Attribute {} = {}'.format(self._att_idx, branch)

    def get_atts_test_depends_on(self):
        return [self._att_idx]
