from skmultiflow.trees.attribute_class_observer import AttributeClassObserver
from skmultiflow.trees.numeric_attribute_binary_test import \
    NumericAttributeBinaryTest
from skmultiflow.trees.attribute_split_suggestion import \
    AttributeSplitSuggestion
import numpy as np


class NumericAttributeRegressionObserverMultiTarget(AttributeClassObserver):
    """iSoup-Tree's Extended Binary Search Tree (E-BST)

    This class implements the Extended Binary Search Tree (E-BST)
    structure, using the variant employed by Osojnik et al. [1]_ in the
    iSoup-Tree algorithm. This structure is employed to observe the target
    space distribution.

    In this variant, only the left branch statistics are stored.

    References:
    .. [1] Osojnik, Aljaž. 2017. Structured output prediction on Data
       Streams (Doctoral Dissertation). Retrieved from:
       http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf
    """

    class Node:
        def __init__(self, att_val, target):
            self.att_val = att_val

            self.k = 1
            self.sum_target = target.copy()
            self.sum_sq_target = target * target

            self._left = None
            self._right = None

        # Incremental implementation of the insert method. Avoiding unecessary
        # stack tracing must decrease memory costs
        def insert_value(self, att_val, target):
            current = self
            antecedent = None

            while current is not None:
                antecedent = current
                if att_val == current.att_val:
                    current.k += 1
                    current.sum_target += target
                    current.sum_sq_target += target * target
                    return
                elif att_val < current.att_val:
                    current.k += 1
                    current.sum_target += target
                    current.sum_sq_target += target * target

                    current = current._left
                    is_right = False
                else:
                    current = current._right
                    is_right = True

            # Value was not yet added to the tree
            if is_right:
                antecedent._right = \
                    NumericAttributeRegressionObserverMultiTarget.\
                    Node(att_val, target)
            else:
                antecedent._left = \
                    NumericAttributeRegressionObserverMultiTarget.\
                    Node(att_val, target)

    def __init__(self):
        super().__init__()
        self._root = None

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = NumericAttributeRegressionObserverMultiTarget.\
                    Node(att_val, class_val)
            else:
                self._root.insert_value(att_val, class_val)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist,
                                            att_idx, binary_only=True):
        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._att_idx = att_idx

        self._aux_k = 0
        self._aux_sum = np.zeros_like(pre_split_dist[1])
        self._aux_sq_sum = np.zeros_like(pre_split_dist[2])

        candidate = AttributeSplitSuggestion(None, [{}], -float('inf'))

        best_split = self._find_best_split(self._root, candidate)

        return best_split

    def _find_best_split(self, node, candidate):
        if node._left is not None:
            candidate = self._find_best_split(node._left, candidate)
        # Left post split distribution
        left_dist = {}
        left_dist[0] = node.k + self._aux_k
        left_dist[1] = node.sum_target + self._aux_sum
        left_dist[2] = node.sum_sq_target + self._aux_sq_sum

        # The right split distribution is calculated as the difference
        # between the total distribution (pre split distribution) and
        # the left distribution
        right_dist = {}
        right_dist[0] = self._pre_split_dist[0] - left_dist[0]
        right_dist[1] = self._pre_split_dist[1] - left_dist[1]
        right_dist[2] = self._pre_split_dist[2] - left_dist[2]

        post_split_dists = [left_dist, right_dist]

        merit = self._criterion.get_merit_of_split(self._pre_split_dist,
                                                   post_split_dists)
        if merit > candidate.merit:
            num_att_binary_test = NumericAttributeBinaryTest(self._att_idx,
                                                             node.att_val,
                                                             True)
            candidate = AttributeSplitSuggestion(num_att_binary_test,
                                                 post_split_dists, merit)

        if node._right is not None:
            self._aux_k += node.k
            self._aux_sum += node.sum_target
            self._aux_sq_sum += node.sum_sq_target

            right_candidate = self._find_best_split(node._right, candidate)

            if right_candidate.merit > candidate.merit:
                candidate = right_candidate

            self._aux_k -= node.k
            self._aux_sum -= node.sum_target
            self._aux_sq_sum -= node.sum_sq_target

        return candidate
