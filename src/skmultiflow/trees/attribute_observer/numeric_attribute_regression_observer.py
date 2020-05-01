import numpy as np
from skmultiflow.trees.attribute_observer import AttributeClassObserver
from skmultiflow.trees.attribute_test import NumericAttributeBinaryTest
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion


class NumericAttributeRegressionObserver(AttributeClassObserver):
    """iSoup-Tree's Extended Binary Search Tree (E-BST)

    This class implements the Extended Binary Search Tree (E-BST)
    structure, using the variant employed by Osojnik et al. [1]_ in the
    iSOUP-Tree algorithm. This structure is employed to observe the target
    space distribution.

    In this variant, only the left branch statistics are stored.

    References
    ----------
    .. [1] Osojnik, Aljaž. 2017. Structured output prediction on Data
       Streams (Doctoral Dissertation). Retrieved from:
       http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf
    """

    class Node:
        def __init__(self, att_val, target, weight):
            self.att_val = att_val

            self.sum_weight = weight
            self.sum_target = weight * target
            self.sum_sq_target = weight * target * target

            self._left = None
            self._right = None

        # Incremental implementation of the insert method. Avoiding unecessary
        # stack tracing must decrease memory costs
        def insert_value(self, att_val, target, weight):
            current = self
            antecedent = None

            while current is not None:
                antecedent = current
                if att_val == current.att_val:
                    current.sum_weight += weight
                    current.sum_target += weight * target
                    current.sum_sq_target += weight * target * target
                    return
                elif att_val < current.att_val:
                    current.sum_weight += weight
                    current.sum_target += weight * target
                    current.sum_sq_target += weight * target * target

                    current = current._left
                    is_right = False
                else:
                    current = current._right
                    is_right = True

            # Value was not yet added to the tree
            if is_right:
                antecedent._right = NumericAttributeRegressionObserver.Node(
                    att_val, target, weight
                )
            else:
                antecedent._left = NumericAttributeRegressionObserver.Node(
                    att_val, target, weight
                )

    def __init__(self):
        super().__init__()
        self._root = None

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = NumericAttributeRegressionObserver.Node(att_val, class_val, weight)
            else:
                self._root.insert_value(att_val, class_val, weight)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        raise NotImplementedError

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist,
                                            att_idx, binary_only=True):
        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._att_idx = att_idx

        self._aux_sum_weight = 0

        # Handles both single-target and multi-target tasks
        if np.ndim(pre_split_dist[1]) == 0:
            self._aux_sum = 0.0
            self._aux_sum_sq = 0.0
        else:
            self._aux_sum = np.zeros_like(pre_split_dist[1])
            self._aux_sum_sq = np.zeros_like(pre_split_dist[2])

        candidate = AttributeSplitSuggestion(None, [{}], -float('inf'))

        best_split = self._find_best_split(self._root, candidate)

        # Reset auxiliary variables
        self._criterion = None
        self._pre_split_dist = None
        self._att_idx = None
        self._aux_sum_weight = None
        self._aux_sum = None
        self._aux_sum_sq = None

        return best_split

    def _find_best_split(self, node, candidate):
        if node._left is not None:
            candidate = self._find_best_split(node._left, candidate)
        # Left post split distribution
        left_dist = {}
        left_dist[0] = node.sum_weight + self._aux_sum_weight
        left_dist[1] = node.sum_target + self._aux_sum
        left_dist[2] = node.sum_sq_target + self._aux_sum_sq

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
            self._aux_sum_weight += node.sum_weight
            self._aux_sum += node.sum_target
            self._aux_sum_sq += node.sum_sq_target

            right_candidate = self._find_best_split(node._right, candidate)

            if right_candidate.merit > candidate.merit:
                candidate = right_candidate

            self._aux_sum_weight -= node.sum_weight
            self._aux_sum -= node.sum_target
            self._aux_sum_sq -= node.sum_sq_target

        return candidate

    def remove_bad_splits(self, criterion, last_check_ratio, last_check_sdr, last_check_e,
                          pre_split_dist):

        # Auxiliary variables
        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._last_check_ratio = last_check_ratio
        self._last_check_sdr = last_check_sdr
        self._last_check_e = last_check_e

        self._aux_sum_weight = 0

        # Encompass both the single-target and multi-target cases
        if np.ndim(pre_split_dist[1]) == 0:
            self._aux_sum = 0.0
            self._aux_sum_sq = 0.0
        else:
            self._aux_sum = np.zeros_like(pre_split_dist[1])
            self._aux_sum_sq = np.zeros_like(pre_split_dist[2])

        self._remove_bad_split_nodes(self._root)

        # Reset auxiliary variables
        self._criterion = None
        self._pre_split_dist = None
        self._last_check_ratio = None
        self._last_check_sdr = None
        self._last_check_e = None
        self._aux_sum_weight = None
        self._aux_sum = None
        self._aux_sum_sq = None

    def _remove_bad_split_nodes(self, current_node, parent=None, is_left_child=True):
        is_bad = False

        if current_node._left is not None:
            is_bad = self._remove_bad_split_nodes(current_node._left, current_node, True)
        else:  # Every leaf node is potentially a bad candidate
            is_bad = True

        if is_bad:
            if current_node._right is not None:
                self._aux_sum_weight += current_node.sum_weight
                self._aux_sum += current_node.sum_target
                self._aux_sum_sq += current_node.sum_sq_target

                is_bad = self._remove_bad_split_nodes(current_node._right, current_node, False)

                self._aux_sum_weight -= current_node.sum_weight
                self._aux_sum -= current_node.sum_target
                self._aux_sum_sq -= current_node.sum_sq_target
            else:  # Every leaf node is potentially a bad candidate
                is_bad = True

        if is_bad:
            # Left post split distribution
            left_dist = {}
            left_dist[0] = current_node.sum_weight + self._aux_sum_weight
            left_dist[1] = current_node.sum_target + self._aux_sum
            left_dist[2] = current_node.sum_sq_target + self._aux_sum_sq

            # The right split distribution is calculated as the difference between the total
            # distribution (pre split distribution) and the left distribution
            right_dist = {}
            right_dist[0] = self._pre_split_dist[0] - left_dist[0]
            right_dist[1] = self._pre_split_dist[1] - left_dist[1]
            right_dist[2] = self._pre_split_dist[2] - left_dist[2]

            post_split_dists = [left_dist, right_dist]
            merit = self._criterion.get_merit_of_split(self._pre_split_dist, post_split_dists)
            if (merit / self._last_check_sdr) < (self._last_check_ratio - 2 * self._last_check_e):
                # Remove children nodes
                current_node._left = None
                current_node._right = None

                # Root node
                if parent is None:
                    self._root = None
                else:  # Root node
                    # Remove bad candidate
                    if is_left_child:
                        parent._left = None
                    else:
                        parent._right = None
                return True

        return False
