from collections import Counter

from skmultiflow.trees._attribute_test import NumericAttributeBinaryTest
from skmultiflow.trees._attribute_test import AttributeSplitSuggestion
from .attribute_observer import AttributeObserver


class NumericAttributeClassObserverBinaryTree(AttributeObserver):
    """ Class for observing the class data distribution for a numeric attribute
    using a binary tree. Used in Naive Bayes and decision trees to monitor data
    statistics on leaves.
    """

    class Node:

        def __init__(self, val, label, weight):
            self._class_count_left = {}
            self._class_count_right = {}
            self._left = None
            self._right = None

            self._cut_point = val
            self._class_count_left[label] += weight

        def insert_value(self, val, label, weight):
            if val == self._cut_point:
                self._class_count_left[label] += weight
            elif val < self._cut_point:
                self._class_count_left[label] += weight
                if self._left is None:
                    self._left = NumericAttributeClassObserverBinaryTree.\
                        Node(val, label, weight)
                else:
                    self._left.insert_value(val, label, weight)
            else:
                self._class_count_right[label] += weight
                if self._right is None:
                    self._right = NumericAttributeClassObserverBinaryTree.\
                        Node(val, label, weight)
                else:
                    self._right.insert_value(val, label, weight)

    """
    end of class Node

    """

    def __init__(self):
        super().__init__()
        self._root = None

    def update(self, att_val, class_val, weight):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = NumericAttributeClassObserverBinaryTree.\
                    Node(att_val, class_val, weight)
            else:
                self._root.insert_value(att_val, class_val, weight)

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist,
                                            att_idx, binary_only):

        return self.search_for_best_split_option(
            current_node=self._root,
            current_best_option=None,
            actual_parent_left=None,
            parent_left=None,
            parent_right=None,
            left_child=False,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx
        )

    def search_for_best_split_option(self, current_node, current_best_option,
                                     actual_parent_left, parent_left,
                                     parent_right, left_child, criterion,
                                     pre_split_dist, att_idx):

        if current_node is None:
            return current_best_option
        left_dist = {}
        right_dist = {}

        if parent_left is None:

            left_dist.update(
                dict(
                    Counter(left_dist) +
                    Counter(current_node._class_count_left)
                )
            )

            right_dist.update(
                dict(
                    Counter(right_dist) +
                    Counter(current_node._class_count_right)
                )
            )
        else:
            left_dist.update(
                dict(
                    Counter(left_dist) +
                    Counter(parent_left)
                )
            )

            right_dist.update(
                dict(
                    Counter(right_dist) +
                    Counter(parent_right)
                )
            )

            if left_child:

                """get the exact statistics of the parent value"""
                exact_parent_dist = {}
                exact_parent_dist.update(
                    dict(
                        Counter(exact_parent_dist) +
                        Counter(actual_parent_left)
                    )
                )

                exact_parent_dist.update(
                    dict(
                        Counter(exact_parent_dist) -
                        Counter(current_node._class_count_left)
                    )
                )

                exact_parent_dist.update(
                    dict(
                        Counter(exact_parent_dist) -
                        Counter(current_node._class_count_right)
                    )
                )

                """move the subtrees"""
                left_dist.update(
                    dict(
                        Counter(left_dist) -
                        Counter(current_node._class_count_right)
                    )
                )

                right_dist.update(
                    dict(
                        Counter(right_dist) +
                        Counter(current_node._class_count_right)
                    )
                )

                """move the exact value from the parent"""
                right_dist.update(
                    dict(
                        Counter(right_dist) +
                        Counter(exact_parent_dist)
                    )
                )

                left_dist.update(
                    dict(
                        Counter(left_dist) -
                        Counter(exact_parent_dist)
                    )
                )

            else:
                left_dist.update(
                    dict(
                        Counter(left_dist) +
                        Counter(current_node._class_count_left)
                    )
                )

                right_dist.update(
                    dict(
                        Counter(right_dist) -
                        Counter(current_node._class_count_left)
                    )
                )

        post_split_dists = [left_dist, right_dist]
        merit = criterion.get_merit_of_split(pre_split_dist, post_split_dists)

        if current_best_option is None or merit > current_best_option.merit:
            num_att_binary_test = \
                NumericAttributeBinaryTest(
                    att_idx=att_idx,
                    att_value=current_node._cut_point,
                    equal_passes_test=True
                )

            current_best_option = \
                AttributeSplitSuggestion(
                    split_test=num_att_binary_test,
                    resulting_class_distributions=post_split_dists,
                    merit=merit
                )

        current_best_option = \
            self.search_for_best_split_option(
                current_node=current_node._left,
                current_best_option=current_best_option,
                actual_parent_left=current_node._class_count_left,
                parent_left=post_split_dists[0],
                parent_right=post_split_dists[1],
                left_child=True,
                criterion=criterion,
                pre_split_dist=pre_split_dist,
                att_idx=att_idx
            )

        current_best_option = \
            self.search_for_best_split_option(
                current_node=current_node._right,
                current_best_option=current_best_option,
                actual_parent_left=current_node._class_count_left,
                parent_left=post_split_dists[0],
                parent_right=post_split_dists[1],
                left_child=False,
                criterion=criterion,
                pre_split_dist=pre_split_dist,
                att_idx=att_idx
            )

        return current_best_option
