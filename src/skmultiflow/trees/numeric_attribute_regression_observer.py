from skmultiflow.trees.numeric_attribute_class_observer_binary_tree import NumericAttributeClassObserverBinaryTree
from skmultiflow.trees.numeric_attribute_binary_test import NumericAttributeBinaryTest
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion


class NumericAttributeRegressionObserver(NumericAttributeClassObserverBinaryTree):
    """ Class for observing the data distribution for a numeric attribute for regression.

    """

    class Node:

        def __init__(self, val, label):
            self._left_statistics = {}
            self._right_statistics = {}
            self._left = None
            self._right = None
            self._cut_point = val
            try:
                self._left_statistics[0] += 1
                self._left_statistics[1] += label
                self._left_statistics[2] += label * label
            except KeyError:
                self._left_statistics[0] = 1
                self._left_statistics[1] = label
                self._left_statistics[2] = label * label
        """
         /**
         * Insert a new value into the tree, updating both the sum of values and
         * sum of squared values arrays
         */

        """

        def insert_value(self, val, label):
            if val == self._cut_point:

                self._left_statistics[0] += 1
                self._left_statistics[1] += label
                self._left_statistics[2] += label * label

            elif val < self._cut_point:

                self._left_statistics[0] += 1
                self._left_statistics[1] += label
                self._left_statistics[2] += label * label
                if self._left is None:
                    self._left = NumericAttributeRegressionObserver.Node(val, label)
                else:
                    self._left.insert_value(val, label)
            else:
                try:
                    self._right_statistics[0] += 1
                    self._right_statistics[1] += label
                    self._right_statistics[2] += label * label
                except KeyError:
                    self._right_statistics[0] = 1
                    self._right_statistics[1] = label
                    self._right_statistics[2] = label * label

                if self._right is None:
                    self._right = NumericAttributeRegressionObserver.Node(val, label)
                else:
                    self._right.insert_value(val, label)

    """
    end of class Node

    """

    def __init__(self):
        super().__init__()
        self._root = None
        self._sum_total_left = 0.0
        self._sum_total_right = 0.0
        self._sum_sq_total_left = 0.0
        self._sum_sq_total_right = 0.0
        self._count_right_total = 0.0
        self._count_left_total = 0.0

    def observe_attribute_class(self, att_val, class_val, weight):

        if att_val is None:
            return

        else:
            if self._root is None:
                self._root = NumericAttributeRegressionObserver.Node(att_val, class_val)
            else:
                self._root.insert_value(att_val, class_val)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):

        self._sum_total_left = 0.0
        self._sum_total_right = pre_split_dist[1]
        self._sum_sq_total_left = 0.0
        self._sum_sq_total_right = pre_split_dist[2]
        self._count_left_total = 0.0
        self._count_right_total = pre_split_dist[0]
        return self.search_for_best_split_option(self._root, None, criterion, att_idx)

    def search_for_best_split_option(self, current_node, current_best_option, criterion, att_idx):
        if current_node is None or self._count_right_total == 0:
            return current_best_option
        if current_node._left is not None:
            current_best_option = self.search_for_best_split_option(current_node._left, current_best_option,
                                                                    criterion, att_idx)
        self._sum_total_left += current_node._left_statistics[1]
        self._sum_total_right -= current_node._left_statistics[1]
        self._sum_sq_total_left += current_node._left_statistics[2]
        self._sum_sq_total_right -= current_node._left_statistics[2]
        self._count_right_total -= current_node._left_statistics[0]
        self._count_left_total += current_node._left_statistics[0]

        lhs_dist = {}
        rhs_dist = {}
        lhs_dist[0] = self._count_left_total
        lhs_dist[1] = self._sum_total_left
        lhs_dist[2] = self._sum_sq_total_left
        rhs_dist[0] = self._count_right_total
        rhs_dist[1] = self._sum_total_right
        rhs_dist[2] = self._sum_sq_total_right
        post_split_dists = [lhs_dist, rhs_dist]
        pre_split_dist = [(self._count_left_total + self._count_right_total),
                          (self._sum_total_left + self._sum_total_right),
                          (self._sum_sq_total_left + self._sum_sq_total_right)]

        merit = criterion.get_merit_of_split(pre_split_dist, post_split_dists)

        if current_best_option is None or merit > current_best_option.merit:
            num_att_binary_test = NumericAttributeBinaryTest(att_idx, current_node._cut_point, True)
            current_best_option = AttributeSplitSuggestion(num_att_binary_test, post_split_dists, merit)

        if current_node._right is not None:
            current_best_option = self.search_for_best_split_option(current_node._right, current_best_option, criterion,
                                                                    att_idx)

        self._sum_total_left -= current_node._left_statistics.get(1)
        self._sum_total_right += current_node._left_statistics.get(1)
        self._sum_sq_total_left -= current_node._left_statistics.get(2)
        self._sum_sq_total_right += current_node._left_statistics.get(2)
        self._count_left_total -= current_node._left_statistics.get(0)
        self._count_right_total += current_node._left_statistics.get(0)

        return current_best_option

    def remove_bad_splits(self, criterion, last_check_ratio, last_check_sdr, last_check_e):

        self.remove_bad_split_nodes(criterion, self._root, last_check_ratio, last_check_sdr, last_check_e)

    def remove_bad_split_nodes(self, criterion, current_node, last_check_ratio, last_check_sdr, last_check_e):

        is_bad = False

        if current_node is None:
            return True

        if current_node._left is not None:
            is_bad = self.remove_bad_split_nodes(criterion, current_node._left, last_check_ratio,
                                                 last_check_sdr, last_check_e)
        if current_node._right is not None and is_bad:
            is_bad = self.remove_bad_split_nodes(criterion, current_node._left, last_check_ratio,
                                                 last_check_sdr, last_check_e)
        if is_bad:
            left_stat = [current_node._left_statistics[0], current_node._left_statistics[1],
                         current_node._left_statistics[2]]
            right_stat = [current_node._right_statistics[0], current_node._right_statistics[1],
                          current_node._right_statistics[2]]

            post_split_dists = [left_stat, right_stat]
            pre_split_dist = [(current_node._left_statistics.get(0) + current_node._right_statistics.get(0)),
                               (current_node._left_statistics.get(1) + current_node._right_statistics.get(1)),
                               (current_node._left_statistics.get(2) + current_node._right_statistics.get(2))]
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dists)
            if (merit / last_check_sdr) < (last_check_ratio - (2 * last_check_e)):
                current_node = None
                return True

        return False
