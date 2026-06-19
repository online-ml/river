from __future__ import annotations

import copy
from collections import Counter, defaultdict

from ..utils import BranchFactory
from .base import Splitter


class ExhaustiveSplitter(Splitter):
    """Numeric attribute observer for classification tasks that is based on
    a Binary Search Tree.

    This algorithm[^1] is also referred to as exhaustive attribute observer,
    since it ends up storing all the observations between split attempts[^2].

    This splitter cannot perform probability density estimations, so it does not work well
    when coupled with tree leaves using naive bayes models.

    References
    ----------
    [^1]: Domingos, P. and Hulten, G., 2000, August. Mining high-speed data streams.
    In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery
    and data mining (pp. 71-80).
    [^2]: Pfahringer, B., Holmes, G. and Kirkby, R., 2008, May. Handling numeric attributes in
    hoeffding trees. In Pacific-Asia Conference on Knowledge Discovery and Data Mining
    (pp. 296-307). Springer, Berlin, Heidelberg.
    """

    def __init__(self):
        super().__init__()
        self._root = None

    def update(self, att_val, target_val, w):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = ExhaustiveNode(att_val, target_val, w)
            else:
                self._root.insert_value(att_val, target_val, w)

    def cond_proba(self, att_val, target_val):
        """The underlying data structure used to monitor the input does not allow probability
        density estimations. Hence, it always returns zero for any given input."""
        return 0.0

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        return self._search_for_best_split_option(
            root=self._root,
            criterion=criterion,
            pre_split_dist=pre_split_dist,
            att_idx=att_idx,
        )

    def _search_for_best_split_option(self, root, criterion, pre_split_dist, att_idx):
        # Iterative pre-order traversal. The recursive form blows Python's
        # recursion limit on long streams since the BST can be degenerate.
        # Each stack entry carries the arguments of one recursive call.
        # Push order is right-then-left so the left subtree is processed
        # first when popping (matching the original pre-order behaviour).
        current_best_option = BranchFactory()
        stack: list[tuple] = [(root, None, None, None, False)]

        while stack:
            node, actual_parent_left, parent_left, parent_right, left_child = stack.pop()
            if node is None:
                continue

            left_dist: dict = {}
            right_dist: dict = {}

            if parent_left is None:
                left_dist.update(dict(Counter(left_dist) + Counter(node.class_count_left)))
                right_dist.update(dict(Counter(right_dist) + Counter(node.class_count_right)))
            else:
                left_dist.update(dict(Counter(left_dist) + Counter(parent_left)))
                right_dist.update(dict(Counter(right_dist) + Counter(parent_right)))

                if left_child:
                    # get the exact statistics of the parent value
                    exact_parent_dist: dict = {}
                    exact_parent_dist.update(
                        dict(Counter(exact_parent_dist) + Counter(actual_parent_left))
                    )
                    exact_parent_dist.update(
                        dict(Counter(exact_parent_dist) - Counter(node.class_count_left))
                    )
                    exact_parent_dist.update(
                        dict(Counter(exact_parent_dist) - Counter(node.class_count_right))
                    )

                    # move the subtrees
                    left_dist.update(dict(Counter(left_dist) - Counter(node.class_count_right)))
                    right_dist.update(dict(Counter(right_dist) + Counter(node.class_count_right)))

                    # move the exact value from the parent
                    right_dist.update(dict(Counter(right_dist) + Counter(exact_parent_dist)))
                    left_dist.update(dict(Counter(left_dist) - Counter(exact_parent_dist)))
                else:
                    left_dist.update(dict(Counter(left_dist) + Counter(node.class_count_left)))
                    right_dist.update(dict(Counter(right_dist) - Counter(node.class_count_left)))

            post_split_dists = [left_dist, right_dist]
            merit = criterion.merit_of_split(pre_split_dist, post_split_dists)
            if merit > current_best_option.merit:
                current_best_option = BranchFactory(
                    merit, att_idx, node.cut_point, post_split_dists
                )

            stack.append(
                (
                    node._right,
                    node.class_count_left,
                    post_split_dists[0],
                    post_split_dists[1],
                    False,
                )
            )
            stack.append(
                (node._left, node.class_count_left, post_split_dists[0], post_split_dists[1], True)
            )

        return current_best_option


class ExhaustiveNode:
    def __init__(self, att_val, target_val, w):
        self.class_count_left = defaultdict(float)
        self.class_count_right = defaultdict(float)
        self._left = None
        self._right = None

        self.cut_point = att_val
        self.class_count_left[target_val] += w

    def __deepcopy__(self, memo):
        # Iterative copy: the BST can be deep enough on long streams that the
        # default recursive deepcopy blows Python's recursion limit.
        cls = type(self)
        new_root = cls.__new__(cls)
        memo[id(self)] = new_root
        stack: list[tuple[ExhaustiveNode, ExhaustiveNode]] = [(self, new_root)]
        while stack:
            src, dst = stack.pop()
            dst.class_count_left = copy.deepcopy(src.class_count_left, memo)
            dst.class_count_right = copy.deepcopy(src.class_count_right, memo)
            dst.cut_point = copy.deepcopy(src.cut_point, memo)
            dst._left = None
            dst._right = None
            if src._left is not None:
                child = cls.__new__(cls)
                memo[id(src._left)] = child
                dst._left = child
                stack.append((src._left, child))
            if src._right is not None:
                child = cls.__new__(cls)
                memo[id(src._right)] = child
                dst._right = child
                stack.append((src._right, child))
        return new_root

    def insert_value(self, val, label, w):
        # Iterative descent: a degenerate (monotonically inserted) BST would
        # otherwise blow Python's recursion limit.
        current = self
        while True:
            if val == current.cut_point:
                current.class_count_left[label] += w
                return
            if val < current.cut_point:
                current.class_count_left[label] += w
                if current._left is None:
                    current._left = ExhaustiveNode(val, label, w)
                    return
                current = current._left
            else:
                current.class_count_right[label] += w
                if current._right is None:
                    current._right = ExhaustiveNode(val, label, w)
                    return
                current = current._right
