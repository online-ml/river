import functools

from river.stats import Var
from river.utils import VectorDict

from .._attribute_test import NumericAttributeBinaryTest
from .._attribute_test import AttributeSplitSuggestion
from .attribute_observer import AttributeObserver


class NumericAttributeRegressionObserver(AttributeObserver):
    """iSOUP-Tree's Extended Binary Search Tree (E-BST).

    This class implements the Extended Binary Search Tree (E-BST)
    structure, using the variant employed by Osojnik et al. [^1] in the
    iSOUP-Tree algorithm. This structure is employed to observe the target
    space distribution.

    In this variant, only the left branch statistics are stored.

    References
    ----------
    [^1]: [Osojnik, Aljaž. 2017. Structured output prediction on Data Streams
    (Doctoral Dissertation)](http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf)
    """

    class Node:
        def __init__(self, att_val, target, sample_weight):
            self.att_val = att_val

            if isinstance(target, dict):
                self.estimator = VectorDict(default_factory=functools.partial(Var))
                self._update_estimator = self._update_estimator_multivariate
            else:
                self.estimator = Var()
                self._update_estimator = self._update_estimator_univariate

            self._update_estimator(self, target, sample_weight)

            self._left = None
            self._right = None

        @staticmethod
        def _update_estimator_univariate(node, target, sample_weight):
            node.estimator.update(target, sample_weight)

        @staticmethod
        def _update_estimator_multivariate(node, target, sample_weight):
            for t in target:
                node.estimator[t].update(target[t], sample_weight)

        # Incremental implementation of the insert method. Avoiding unnecessary
        # stack tracing must decrease memory costs
        def insert_value(self, att_val, target, sample_weight):
            current = self
            antecedent = None

            while current is not None:
                antecedent = current
                if att_val == current.att_val:
                    self._update_estimator(current, target, sample_weight)
                    return
                elif att_val < current.att_val:
                    self._update_estimator(current, target, sample_weight)

                    current = current._left
                    is_right = False
                else:
                    current = current._right
                    is_right = True

            # Value was not yet added to the tree
            if is_right:
                antecedent._right = NumericAttributeRegressionObserver.Node(
                    att_val, target, sample_weight
                )
            else:
                antecedent._left = NumericAttributeRegressionObserver.Node(
                    att_val, target, sample_weight
                )

    def __init__(self):
        super().__init__()
        self._root = None

    def update(self, att_val, target_val, sample_weight):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = NumericAttributeRegressionObserver.Node(
                    att_val, target_val, sample_weight
                )
            else:
                self._root.insert_value(att_val, target_val, sample_weight)

        return self

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        raise NotImplementedError

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only=True):
        candidate = AttributeSplitSuggestion(None, [{}], -float("inf"))

        if self._root is None:
            return candidate

        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._att_idx = att_idx

        # Handles both single-target and multi-target tasks
        if isinstance(pre_split_dist, VectorDict):
            self._aux_estimator = VectorDict(default_factory=functools.partial(Var))
        else:
            self._aux_estimator = Var()

        best_split = self._find_best_split(self._root, candidate)

        # Delete auxiliary variables
        del self._criterion
        del self._pre_split_dist
        del self._att_idx
        del self._aux_estimator

        return best_split

    def _find_best_split(self, node, candidate):
        if node._left is not None:
            candidate = self._find_best_split(node._left, candidate)
        # Left post split distribution
        left_dist = node.estimator + self._aux_estimator

        # The right split distribution is calculated as the difference between the total
        # distribution (pre split distribution) and the left distribution
        right_dist = self._pre_split_dist - left_dist

        post_split_dists = [left_dist, right_dist]

        merit = self._criterion.merit_of_split(self._pre_split_dist, post_split_dists)
        if merit > candidate.merit:
            num_att_binary_test = NumericAttributeBinaryTest(self._att_idx, node.att_val, True)
            candidate = AttributeSplitSuggestion(num_att_binary_test, post_split_dists, merit)

        if node._right is not None:
            self._aux_estimator += node.estimator

            right_candidate = self._find_best_split(node._right, candidate)

            if right_candidate.merit > candidate.merit:
                candidate = right_candidate

            self._aux_estimator -= node.estimator

        return candidate

    def remove_bad_splits(
        self, criterion, last_check_ratio, last_check_vr, last_check_e, pre_split_dist
    ):
        """Remove bad splits.

        Based on FIMT-DD's [^1] procedure to remove bad split candidates from the E-BST. This
        mechanism is triggered every time a split attempt fails. The rationale is to remove
        points whose split merit is much worse than the best candidate overall (for which the
        growth decision already failed).

        Let $m_1$ be the merit of the best split point and $m_2$ be the merit of the
        second best split candidate. The ratio $r = m_2/m_1$ along with the Hoeffding bound
        ($\\epsilon$) are used to decide upon creating a split. A split occurs when
        $r < 1 - \\epsilon$. A split candidate, with merit $m_i$, is considered badr
        if $m_i / m_1 < r - 2\\epsilon$. The rationale is the following: if the merit ratio
        for this point is smaller than the lower bound of $r$, then the true merit of that
        split relative to the best one is small. Hence, this candidate can be safely removed.

        To avoid excessive and costly manipulations of the E-BST to update the stored statistics,
        only the nodes whose children are all bad split points are pruned, as defined in [^1].

        Parameters
        ----------
        criterion
            The split criterion used by the regression tree.
        last_check_ratio
            The ratio between the merit of the second best split candidate and the merit of the
            best split candidate observed in the last failed split attempt.
        last_check_vr
            The merit (variance reduction) of the best split candidate observed in the last
            failed split attempt.
        last_check_e
            The Hoeffding bound value calculated in the last failed split attempt.
        pre_split_dist
            The complete statistics of the target observed in the leaf node.

        References
        ----------
        [^1]: Ikonomovska, E., Gama, J., & Džeroski, S. (2011). Learning model trees from evolving
        data streams. Data mining and knowledge discovery, 23(1), 128-168.
        """

        if self._root is None:
            return

        # Auxiliary variables
        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._last_check_ratio = last_check_ratio
        self._last_check_vr = last_check_vr
        self._last_check_e = last_check_e

        # Handles both single-target and multi-target tasks
        if isinstance(pre_split_dist, VectorDict):
            self._aux_estimator = VectorDict(default_factory=functools.partial(Var))
        else:
            self._aux_estimator = Var()

        self._remove_bad_split_nodes(self._root)

        # Delete auxiliary variables
        del self._criterion
        del self._pre_split_dist
        del self._last_check_ratio
        del self._last_check_vr
        del self._last_check_e
        del self._aux_estimator

    def _remove_bad_split_nodes(self, current_node, parent=None, is_left_child=True):
        is_bad = False

        if current_node._left is not None:
            is_bad = self._remove_bad_split_nodes(current_node._left, current_node, True)
        else:  # Every leaf node is potentially a bad candidate
            is_bad = True

        if is_bad:
            if current_node._right is not None:
                self._aux_estimator += current_node.estimator

                is_bad = self._remove_bad_split_nodes(current_node._right, current_node, False)

                self._aux_estimator -= current_node.estimator
            else:  # Every leaf node is potentially a bad candidate
                is_bad = True

        if is_bad:
            # Left post split distribution
            left_dist = current_node.estimator + self._aux_estimator

            # The right split distribution is calculated as the difference between the total
            # distribution (pre split distribution) and the left distribution
            right_dist = self._pre_split_dist - left_dist

            post_split_dists = [left_dist, right_dist]
            merit = self._criterion.merit_of_split(self._pre_split_dist, post_split_dists)
            if (merit / self._last_check_vr) < (self._last_check_ratio - 2 * self._last_check_e):
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
