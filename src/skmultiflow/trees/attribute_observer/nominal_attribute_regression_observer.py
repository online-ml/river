import numpy as np
from skmultiflow.trees.attribute_observer import AttributeClassObserver
from skmultiflow.trees.attribute_test import NominalAttributeBinaryTest
from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion


class NominalAttributeRegressionObserver(AttributeClassObserver):
    """ Class for observing the data distribution for a nominal attribute for regression.

    """

    class Node:

        _statistics = {}
        _child = None

        def __init__(self, val, label, weight=None):
            self._cut_point = val
            self._statistics[0] = 1
            self._statistics[1] = label
            self._statistics[2] = label * label

        """
         /**
         * Insert a new value into the tree, updating both the sum of values and
         * sum of squared values arrays
         */

        """

        def insert_value(self, val, label):
            if val == self._cut_point:

                self._statistics[0] += 1
                self._statistics[1] += label
                self._statistics[2] += label * label

            else:

                if self._child is None:
                    self._child = NominalAttributeRegressionObserver.Node(val, label)
                else:
                    self._child.insert_value(val, label)

    """
    end of class Node

    """

    def __init__(self):
        super().__init__()
        self._root = None
        self._sum_one = 0.0
        self._sum_rest = 0.0
        self._sum_sq_rest = 0.0
        self._sum_sq_one = 0.0
        self._count_one = 0.0
        self._count_rest = 0.0
        self._sum_total = 0.0
        self._sum_sq_total = 0.0
        self._count = 0.0
        self._binary_only = True
        self._number_of_possible_values = 0

    def observe_attribute_class(self, att_val, class_val, weight):

        if att_val is None:
            return

        else:
            if self._root is None:
                self._number_of_possible_values = 1
                self._root = NominalAttributeRegressionObserver.Node(att_val, class_val, weight)
            else:
                self._root.insert_value(att_val, class_val)


    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):

        self._sum_one = 0.0
        self._sum_rest = 0.0
        self._sum_sq_rest = 0.0
        self._sum_sq_one = 0.0
        self._count_one = 0.0
        self._count_rest = 0.0
        self._sum_total = pre_split_dist[1]
        self._sum_sq_total = pre_split_dist[2]
        self._count = pre_split_dist[0]
        self._binary_only = binary_only
        if self._binary_only:
            return self.search_for_best_binary_split_option(self._root, None, criterion, att_idx)
        else:
            return self.search_for_best_multiway_split_option(self._root, None, criterion, att_idx)

    def search_for_best_binary_split_option(self, current_node, current_best_option, criterion, att_idx):
        if current_node is None or self._count_rest == 0:
            return current_best_option

        if current_node._child is not None:
            current_best_option = self.search_for_best_binary_split_option(current_node._left, current_best_option,
                                                                    criterion, att_idx)
        self._sum_one = current_node._statistics.get(1)
        self._sum_rest = self._sum_total - self._sum_one
        self._sum_sq_one = current_node._statistics.get(2)
        self._sum_sq_rest = self._sum_sq_total - self._sum_sq_one
        self._count_one = current_node._statistics.get(0)
        self._count_rest = self._count - self._count_one

        one_dict = {self._count_one, self._sum_one, self._sum_sq_one}
        rest_dict = {self._count_rest, self._sum_rest, self._sum_sq_rest}
        post_split_dists = [one_dict, rest_dict]
        pre_split_dist = [{self._count, self._sum_total, self._sum_sq_total}]

        merit = criterion.get_merit_of_split(pre_split_dist, post_split_dists)

        if current_best_option is None or merit > current_best_option.merit:
            nom_att_binary_test = NominalAttributeBinaryTest(att_idx, current_node._cut_point)
            current_best_option = AttributeSplitSuggestion(nom_att_binary_test, post_split_dists, merit)

        return current_best_option

# TODO Also consider nominal attributes
    def search_for_best_multiway_split_option(self, current_node, current_best_option, criterion, att_idx):
        post_split_dists = np.zeros([self._number_of_possible_values, 3])
        if current_node is None or self._count_rest == 0:
                return current_best_option
        for i in range(self._number_of_possible_values):
            post_split_dists[i, 0] = current_node._statistics.get(0)
            post_split_dists[i, 1] = current_node._statistics.get(1)
            post_split_dists[i, 2] = current_node._statistics.get(2)
            current_node = current_node._child

        pre_split_dist = [{self._count, self._sum_total, self._sum_sq_total}]
        merit = criterion.get_merit_of_split(pre_split_dist, post_split_dists)
        if current_best_option is None or merit > current_best_option.merit:
            nom_att_mutliway_test = NominalAttributeMultiwayTest(att_idx)
            current_best_option = AttributeSplitSuggestion(nom_att_mutliway_test, post_split_dists, merit)

        return current_best_option
