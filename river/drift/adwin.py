import numpy as np

from river.base import DriftDetector


class ADWIN(DriftDetector):
    r"""Adaptive Windowing method for concept drift detection.

    ADWIN (ADaptive WINdowing) is a popular drift detection method with
    mathematical guarantees. ADWIN efficiently keeps a variable-length window
    of recent items; such that it holds that there has no been change in the
    data distribution. This window is further divided into two sub-windows
    $(W_0, W_1)$ used to determine if a change has happened. ADWIN compares
    the average of $W_0$ and $W_1$ to confirm that they correspond to the
    same distribution. Concept drift is detected if the distribution equality
    no longer holds. Upon detecting a drift, $W_0$ is replaced by $W_1$ and a
    new $W_1$ is initialized. ADWIN uses a confidence value
    $\delta=\in(0,1)$ to determine if the two sub-windows correspond to the
    same distribution.

    **Input**: `value` can be any numeric value related to the definition of
    concept change for the data analyzed. For example, using 0's or 1's
    to track drift in a classifier's performance as follows:

    - 0: Means the learners prediction was wrong

    - 1: Means the learners prediction was correct

    Parameters
    ----------
    delta
        Confidence value.

    Examples
    --------
    >>> import numpy as np
    >>> from river.drift import ADWIN
    >>> np.random.seed(12345)

    >>> adwin = ADWIN()

    >>> # Simulate a data stream composed by two data distributions
    >>> data_stream = np.concatenate((np.random.randint(2, size=1000),
    ...                               np.random.randint(4, high=8, size=1000)))

    >>> # Update drift detector and verify if change is detected
    >>> for i, val in enumerate(data_stream):
    ...     in_drift, in_warning = adwin.update(val)
    ...     if in_drift:
    ...         print(f"Change detected at index {i}, input value: {val}")
    Change detected at index 1023, input value: 5
    Change detected at index 1055, input value: 7
    Change detected at index 1087, input value: 5

    References
    ----------
    [^1]: Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing." In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

    """

    MAX_BUCKETS = 5

    def __init__(self, delta=0.002):
        super().__init__()
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_row = 0
        self.list_row_bucket = None
        self._total = 0
        self._variance = 0
        self._width = 0
        self.bucket_number = 0

        self.__init_buckets()

        # other default values
        self.mint_min_window_longitude = 10

        self.mdbl_delta = 0.002
        self.mint_time = 0
        self.mdbl_width = 0

        self.detect = 0
        self._n_detections = 0
        self.detect_twice = 0
        self.mint_clock = 32

        self.bucket_num_max = 0
        self.mint_min_window_length = 5
        super().reset()

    def reset(self):
        """Reset the change detector."""
        self.__init__(delta=self.delta)

    @property
    def _bucket_used_bucket(self):
        return self.bucket_num_max

    @property
    def width(self):
        """Window size"""
        return self._width

    @property
    def n_detections(self):
        return self._n_detections

    @property
    def variance(self):
        return self._variance / self._width

    @property
    def estimation(self):
        """Error estimation"""
        if self._width == 0:
            return 0
        return self._total / self._width

    def __init_buckets(self):
        """Initialize the bucket's List and statistics

        Set all statistics to 0 and create a new bucket List.

        """
        self.list_row_bucket = List()
        self.last_bucket_row = 0
        self._total = 0
        self._variance = 0
        self._width = 0
        self.bucket_number = 0

    def update(self, value):
        """Update the change detector with a single data point.

        Apart from adding the element value to the window, by inserting it in
        the correct bucket, it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.

        Parameters
        ----------
        value
            Input value

        Returns
        -------
        tuple
            A tuple (drift, warning) where its elements indicate if a drift or a warning is
            detected.

        """
        self._width += 1
        self.__insert_element_bucket(0, value, self.list_row_bucket.first)
        incremental_variance = 0

        if self._width > 1:
            incremental_variance = (
                (self._width - 1)
                * (value - self._total / (self._width - 1))
                * (value - self._total / (self._width - 1))
                / self._width
            )

        self._variance += incremental_variance
        self._total += value
        self.__compress_buckets()

        return self._detect_change()

    def __insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

    @staticmethod
    def _bucket_size(row):
        return np.power(2, row)

    def _delete_element(self) -> int:
        """Delete an item from the bucket list.

        Deletes the last item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        The bucket size from the updated bucket

        """
        node = self.list_row_bucket.last
        n1 = self._bucket_size(self.last_bucket_row)
        self._width -= n1
        self._total -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = node.get_variance(0) + n1 * self._width * (
            u1 - self._total / self._width
        ) * (u1 - self._total / self._width) / (n1 + self._width)
        self._variance -= incremental_variance
        node.remove_bucket()
        self.bucket_number -= 1

        if node.bucket_size_row == 0:
            self.list_row_bucket.remove_from_tail()
            self.last_bucket_row -= 1

        return n1

    def __compress_buckets(self):
        cursor = self.list_row_bucket.first
        i = 0
        while cursor is not None:
            k = cursor.bucket_size_row
            if k == self.MAX_BUCKETS + 1:
                next_node = cursor.get_next_item()
                if next_node is None:
                    self.list_row_bucket.add_to_tail()
                    next_node = cursor.get_next_item()
                    self.last_bucket_row += 1
                n1 = self._bucket_size(i)
                n2 = self._bucket_size(i)
                u1 = cursor.get_total(0) / n1
                u2 = cursor.get_total(1) / n2
                incremental_variance = n1 * n2 * ((u1 - u2) * (u1 - u2)) / (n1 + n2)
                next_node.insert_bucket(
                    cursor.get_total(0) + cursor.get_total(1),
                    cursor.get_variance(1) + incremental_variance,
                )
                self.bucket_number += 1
                cursor.compress_bucket_row(2)

                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break

            cursor = cursor.get_next_item()
            i += 1

    def _detect_change(self) -> bool:
        """Detects concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from
        Time-Changing Data with Adaptive Windowing'. The general idea is to keep
        statistics from a window of variable size while detecting concept drift.

        This function is responsible for analysing different cutting points in
        the sliding window, to verify if there is a significant change in concept.

        Returns
        -------
        Whether change was detected or not

        Notes
        -----
        If change was detected, one should verify the new window size, by reading the width
        property.

        """
        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (
            self.width > self.mint_min_window_longitude
        ):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self._width
                u0 = 0
                u1 = self._total
                v0 = 0
                v1 = self._variance
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket.last
                i = self.last_bucket_row

                while (not bln_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row - 1):
                        n2 = self._bucket_size(i)
                        u2 = cursor.get_total(k)

                        if n0 > 0:
                            v0 += cursor.get_variance(k) + 1.0 * n0 * n2 * (u0 / n0 - u2 / n2) * (
                                u0 / n0 - u2 / n2
                            ) / (n0 + n2)

                        if n1 > 0:
                            v1 -= cursor.get_variance(k) + 1.0 * n1 * n2 * (u1 / n1 - u2 / n2) * (
                                u1 / n1 - u2 / n2
                            ) / (n1 + n2)

                        n0 += self._bucket_size(i)
                        n1 -= self._bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if (i == 0) and (k == cursor.bucket_size_row - 1):
                            bln_exit = True
                            break

                        abs_value = 1.0 * ((u0 / n0) - (u1 / n1))
                        if (
                            (n1 >= self.mint_min_window_length)
                            and (n0 >= self.mint_min_window_length)
                            and (
                                self.__bln_cut_expression(
                                    n0, n1, u0, u1, v0, v1, abs_value, self.delta
                                )
                            )
                        ):
                            bln_bucket_deleted = True  # noqa: F841
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            if self.width > 0:
                                n0 -= self._delete_element()
                                bln_exit = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1
        self.mdbl_width += self.width
        if bln_change:
            self._n_detections += 1
        self._in_concept_change = bln_change

        return self._in_concept_change, self._in_warning_zone

    def __bln_cut_expression(self, n0, n1, u0, u1, v0, v1, abs_value, delta):
        n = self.width
        dd = np.log(2 * np.log(n) / delta)
        v = self.variance
        m = (1.0 / (n0 - self.mint_min_window_length + 1)) + (
            1.0 / (n1 - self.mint_min_window_length + 1)
        )
        epsilon = np.sqrt(2 * m * v * dd) + 1.0 * 2 / 3 * dd * m
        return np.absolute(abs_value) > epsilon


class List(object):
    """A linked list object for ADWIN algorithm.

    Used for storing ADWIN's bucket list. Is composed of Item objects.
    Acts as a linked list, where each element points to its predecessor
    and successor.

    """

    def __init__(self):
        super().__init__()
        self._count = None
        self._first = None
        self._last = None
        self.reset()
        self.add_to_head()

    def reset(self):
        self._count = 0
        self._first = None
        self._last = None

    def add_to_head(self):
        self._first = Item(self._first, None)
        if self._last is None:
            self._last = self._first

    def remove_from_head(self):
        self._first = self._first.get_next_item()
        if self._first is not None:
            self._first.set_previous(None)
        else:
            self._last = None
        self._count -= 1

    def add_to_tail(self):
        self._last = Item(None, self._last)
        if self._first is None:
            self._first = self._last
        self._count += 1

    def remove_from_tail(self):
        self._last = self._last.get_previous()
        if self._last is not None:
            self._last.set_next_item(None)
        else:
            self._first = None
        self._count -= 1

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def size(self):
        return self._count


class Item(object):
    """Item to be used by the List object.

    The Item object, alongside the List object, are the two main data
    structures used for storing the relevant statistics for the ADWIN
    algorithm for change detection.

    Parameters
    ----------
    next_item: Item object
        Reference to the next Item in the List
    previous_item: Item object
        Reference to the previous Item in the List

    """

    def __init__(self, next_item=None, previous_item=None):
        super().__init__()
        self.next = next_item
        self.previous = previous_item
        if next_item is not None:
            next_item.previous = self
        if previous_item is not None:
            previous_item.set_next_item(self)
        self.bucket_size_row = None
        self.max_buckets = ADWIN.MAX_BUCKETS
        self.bucket_total = np.zeros(self.max_buckets + 1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets + 1, dtype=float)
        self.reset()

    def reset(self):
        """Reset the algorithm's statistics and window

        Returns
        -------
        ADWIN
            self

        """
        self.bucket_size_row = 0
        for i in range(ADWIN.MAX_BUCKETS + 1):
            self.__clear_buckets(i)

        return self

    def __clear_buckets(self, index):
        self.set_total(0, index)
        self.set_variance(0, index)

    def insert_bucket(self, value, variance):
        new_item = self.bucket_size_row
        self.bucket_size_row += 1
        self.set_total(value, new_item)
        self.set_variance(variance, new_item)

    def remove_bucket(self):
        self.compress_bucket_row(1)

    def compress_bucket_row(self, num_deleted=1):
        for i in range(num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.bucket_total[i - num_deleted] = self.bucket_total[i]
            self.bucket_variance[i - num_deleted] = self.bucket_variance[i]

        for i in range(1, num_deleted + 1):
            self.__clear_buckets(ADWIN.MAX_BUCKETS - i + 1)

        self.bucket_size_row -= num_deleted

    def get_next_item(self):
        return self.next

    def set_next_item(self, next_item):
        self.next = next_item

    def get_previous(self):
        return self.previous

    def set_previous(self, previous):
        self.previous = previous

    def get_total(self, index):
        return self.bucket_total[index]

    def get_variance(self, index):
        return self.bucket_variance[index]

    def set_total(self, value, index):
        self.bucket_total[index] = value

    def set_variance(self, value, index):
        self.bucket_variance[index] = value
