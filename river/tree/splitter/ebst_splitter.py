from __future__ import annotations

import copy
import functools

from river.stats import Var

from ..utils import BranchFactory
from .base import Splitter


class EBSTSplitter(Splitter):
    """iSOUP-Tree's Extended Binary Search Tree (E-BST).

    This class implements the Extended Binary Search Tree[^1] (E-BST)
    structure, using the variant employed by Osojnik et al.[^2] in the
    iSOUP-Tree algorithm. This structure is employed to observe the target
    space distribution.

    Proposed along with Fast Incremental Model Tree with Drift Detection[^1] (FIMT-DD), E-BST was
    the first attribute observer (AO) proposed for incremental Hoeffding Tree regressors. This
    AO works by storing all observations between splits in an extended binary search tree
    structure. E-BST stores the input feature realizations and statistics of the target(s) that
    enable calculating the split heuristic at any time. To alleviate time and memory costs, E-BST
    implements a memory management routine, where the worst split candidates are pruned from the
    binary tree.

    In this variant, only the left branch statistics are stored and the complete split-enabling
    statistics are calculated with an in-order traversal of the binary search tree.

    References
    ----------
    [^1]: Ikonomovska, E., Gama, J., & Džeroski, S. (2011). Learning model trees from evolving
        data streams. Data mining and knowledge discovery, 23(1), 128-168.
    [^2]: [Osojnik, Aljaž. 2017. Structured output prediction on Data Streams
    (Doctoral Dissertation)](http://kt.ijs.si/theses/phd_aljaz_osojnik.pdf)

    """

    def __init__(self):
        super().__init__()
        self._root = None

    @property
    def is_target_class(self) -> bool:
        return False

    def update(self, att_val, target_val, w):
        if att_val is None:
            return
        else:
            if self._root is None:
                self._root = EBSTNode(att_val, target_val, w)
            else:
                self._root.insert_value(att_val, target_val, w)

    def cond_proba(self, att_val, target_val):
        """Not implemented in regression splitters."""
        raise NotImplementedError

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only=True):
        candidate = BranchFactory()

        if self._root is None:
            return candidate

        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._att_idx = att_idx

        # Import VectorDict here to prevent circular import of river.utils
        from river.utils import VectorDict

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
        # Iterative in-order traversal. The recursive form blows Python's
        # recursion limit on long streams (the BST can be degenerate or deep).
        # We use an operation stack so we can run code after a node's right
        # subtree finishes — needed to undo `_aux_estimator += node.estimator`.
        ops: list[tuple[str, object]] = [("descend", node)]
        while ops:
            action, n = ops.pop()
            if action == "descend":
                # Push in reverse of execution order; the stack pops in LIFO order.
                if n._right is not None:  # type: ignore[union-attr]
                    ops.append(("sub_aux", n))
                    ops.append(("descend", n._right))  # type: ignore[union-attr]
                    ops.append(("add_aux", n))
                ops.append(("process", n))
                if n._left is not None:  # type: ignore[union-attr]
                    ops.append(("descend", n._left))  # type: ignore[union-attr]
            elif action == "add_aux":
                self._aux_estimator += n.estimator  # type: ignore[union-attr]
            elif action == "sub_aux":
                self._aux_estimator -= n.estimator  # type: ignore[union-attr]
            else:  # "process"
                left_dist = n.estimator + self._aux_estimator  # type: ignore[union-attr]
                right_dist = self._pre_split_dist - left_dist
                post_split_dists = [left_dist, right_dist]
                merit = self._criterion.merit_of_split(self._pre_split_dist, post_split_dists)
                if merit > candidate.merit:
                    candidate = BranchFactory(
                        merit,
                        self._att_idx,
                        n.att_val,  # type: ignore[union-attr]
                        post_split_dists,
                    )
        return candidate

    def remove_bad_splits(
        self,
        criterion,
        last_check_ratio: float,
        last_check_vr: float,
        last_check_e: float,
        pre_split_dist: list | dict,
    ):
        """Remove bad splits.

        Based on FIMT-DD's procedure to remove bad split candidates from the E-BST. This
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
        only the nodes whose children are all bad split points are pruned, as defined in the original paper.

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

        # Import VectorDict here to prevent circular import of river.utils
        from river.utils import VectorDict

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
        # Iterative post-order traversal: we need both children's `is_bad`
        # before we can evaluate the current node, and we must undo
        # `_aux_estimator += node.estimator` after each right subtree.
        PHASE_LEFT, PHASE_RIGHT, PHASE_EVAL = 0, 1, 2

        # Frame: [node, parent, is_left_child, phase, aux_added]
        # `aux_added` tracks whether this frame added node.estimator to
        # `_aux_estimator` when descending right — needed because the right
        # child may prune itself (setting node._right = None) before this
        # frame reaches PHASE_EVAL, so we can't rely on `node._right` to
        # decide whether to undo the addition.
        stack: list[list] = [[current_node, parent, is_left_child, PHASE_LEFT, False]]
        # Carries the most recently completed child's return value.
        child_is_bad = False

        while stack:
            frame = stack[-1]
            node, p, ilc, phase, _ = frame

            if phase == PHASE_LEFT:
                if node._left is not None:
                    frame[3] = PHASE_RIGHT
                    stack.append([node._left, node, True, PHASE_LEFT, False])
                    continue
                # Leaf on the left side → treated as bad in the original.
                child_is_bad = True
                frame[3] = PHASE_RIGHT
                continue

            if phase == PHASE_RIGHT:
                if not child_is_bad:
                    # Left subtree was clean → this subtree is clean too;
                    # short-circuit without evaluating right or self.
                    stack.pop()
                    child_is_bad = False
                    continue
                if node._right is not None:
                    self._aux_estimator += node.estimator
                    frame[3] = PHASE_EVAL
                    frame[4] = True
                    stack.append([node._right, node, False, PHASE_LEFT, False])
                    continue
                # Leaf on the right side → still bad.
                child_is_bad = True
                frame[3] = PHASE_EVAL
                continue

            # PHASE_EVAL: both subtrees have been processed (or were absent).
            if frame[4]:
                self._aux_estimator -= node.estimator

            is_bad = child_is_bad
            stack.pop()

            if is_bad:
                left_dist = node.estimator + self._aux_estimator
                right_dist = self._pre_split_dist - left_dist
                post_split_dists = [left_dist, right_dist]
                merit = self._criterion.merit_of_split(self._pre_split_dist, post_split_dists)
                if (merit / self._last_check_vr) < (
                    self._last_check_ratio - 2 * self._last_check_e
                ):
                    node._left = None
                    node._right = None
                    if p is None:
                        self._root = None
                    elif ilc:
                        p._left = None
                    else:
                        p._right = None
                    child_is_bad = True
                    continue

            child_is_bad = False

        return child_is_bad


class EBSTNode:
    __slots__ = ("att_val", "estimator", "_update_estimator", "_left", "_right")

    def __init__(self, att_val, target_val, w):
        self.att_val = att_val

        if isinstance(target_val, dict):
            # Import VectorDict here to prevent circular import of river.utils
            from river.utils import VectorDict

            self.estimator = VectorDict(default_factory=functools.partial(Var))
            self._update_estimator = self._update_estimator_multivariate
        else:
            self.estimator = Var()
            self._update_estimator = self._update_estimator_univariate

        self._update_estimator(self, target_val, w)

        self._left = None
        self._right = None

    @staticmethod
    def _update_estimator_univariate(node, target, w):
        node.estimator.update(target, w)

    @staticmethod
    def _update_estimator_multivariate(node, target, w):
        for t in target:
            node.estimator[t].update(target[t], w)

    def __deepcopy__(self, memo):
        # Iterative copy: the BST can be deep enough on long streams that the
        # default recursive deepcopy (parent -> _left -> deepcopy(child) -> ...)
        # blows Python's recursion limit.
        cls = type(self)
        new_root = cls.__new__(cls)
        memo[id(self)] = new_root
        stack = [(self, new_root)]
        while stack:
            src, dst = stack.pop()
            dst.att_val = copy.deepcopy(src.att_val, memo)
            dst.estimator = copy.deepcopy(src.estimator, memo)
            # The dispatch method is a static reference; rebinding is fine.
            dst._update_estimator = src._update_estimator
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

    # Incremental implementation of the insert method. Avoiding unnecessary
    # stack tracing must decrease memory costs
    def insert_value(self, att_val, target_val, w):
        current = self
        antecedent = None
        is_right = False

        while current is not None:
            antecedent = current
            if att_val == current.att_val:
                self._update_estimator(current, target_val, w)
                return
            elif att_val < current.att_val:
                self._update_estimator(current, target_val, w)

                current = current._left
                is_right = False
            else:
                current = current._right
                is_right = True

        # Value was not yet added to the tree
        if is_right:
            antecedent._right = EBSTNode(att_val, target_val, w)
        else:
            antecedent._left = EBSTNode(att_val, target_val, w)
