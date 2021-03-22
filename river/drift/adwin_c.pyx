# cython: boundscheck=False

from libc.math cimport sqrt, log, fabs, pow
import numpy as np
cimport numpy as np
from collections import deque


cdef class ADWINC:
    """ The helper class for ADWIN

    Parameters
    ----------
    delta
        Confidence value.

    """
    cdef:
        # dict __dict__
        BucketList bucket_list
        double delta, total, variance, total_width, width
        int last_bucket_idx, n_buckets, min_window_len,\
            mint_min_window_length, tick, n_detections,\
            clock, max_n_buckets, detect, detect_twice

    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_idx = 0
        self.bucket_list = None
        self.total = 0.
        self.variance = 0.
        self.width = 0.
        self.n_buckets = 0

        self._init_buckets()

        # other default values
        self.min_window_len = 10

        self.tick = 0
        self.total_width = 0

        self.detect = 0
        self.n_detections = 0
        self.detect_twice = 0
        self.clock = 32

        self.max_n_buckets = 0
        self.mint_min_window_length = 5

    def reset(self):
        """Reset the change detector.
        """
        self.__init__(delta=self.delta)

    def get_delta(self):
        return self.delta

    def get_n_detections(self):
        return self.n_detections

    def get_width(self):
        return self.width

    def get_total(self):
        return self.total

    def get_variance(self):
        return self.variance

    @property
    def variance_in_window(self):
        return self.variance / self.width

    cdef void _init_buckets(self):
        """ Initialize the bucket's list and statistics

        Set all statistics to 0 and create a new bucket List.

        """
        self.bucket_list = BucketList()
        self.last_bucket_idx = 0
        self.total = 0.0
        self.variance = 0.0
        self.width = 0.0
        self.n_buckets = 0

    cpdef bint update(self, double value):
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
        bint
            If True then a change is detected.

        """
        cdef double incremental_variance

        self.width += 1

        self._insert_element_bucket(0, value, self.bucket_list.first)
        incremental_variance = 0.0
        if self.width > 1.0:
            incremental_variance = (
                    (self.width - 1.0)
                    * (value - self.total / (self.width - 1.0))
                    * (value - self.total / (self.width - 1.0))
                    / self.width
            )
        self.variance += incremental_variance
        self.total += value
        self._compress_buckets()

        return self._detect_change()

    cdef void _insert_element_bucket(self, double variance, double value, Item node):
        node.insert_bucket(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

    # @staticmethod
    cdef double _bucket_size(self, int row):
        return pow(2, row)

    cdef double _delete_element(self):
        """Delete an item from the bucket list.

        Deletes the last item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        The bucket size from the updated bucket list

        """
        cdef:
            double n1, u1
            Item node
            double incremental_variance

        node = self.bucket_list.last
        n1 = self._bucket_size(self.last_bucket_idx)
        self.width -= n1
        self.total -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = (
                node.get_variance(0) + n1 * self.width
                * (u1 - self.total / self.width) * (u1 - self.total / self.width)
                / (n1 + self.width)
        )
        self.variance -= incremental_variance
        node.remove_bucket()
        self.n_buckets -= 1

        if node.n_buckets_in_row == 0:
            self.bucket_list.remove_from_tail()
            self.last_bucket_idx -= 1

        return n1

    cdef void _compress_buckets(self):

        cdef:
            int i, k
            double n1, n2, u1, u2, incremental_variance
            Item cursor, next_node

        cursor = self.bucket_list.first
        i = 0
        while cursor is not None:
            k = cursor.n_buckets_in_row
            # If the row is full, merge buckets
            if k == self.MAX_BUCKETS + 1:
                next_node = cursor.get_next_item()
                if next_node is None:
                    self.bucket_list.add_to_tail()
                    next_node = cursor.get_next_item()
                    self.last_bucket_idx += 1
                n1 = self._bucket_size(i)
                n2 = self._bucket_size(i)
                u1 = cursor.get_total(0) / n1
                u2 = cursor.get_total(1) / n2
                incremental_variance = n1 * n2 * ((u1 - u2) * (u1 - u2)) / (n1 + n2)
                next_node.insert_bucket(
                    cursor.get_total(0) + cursor.get_total(1),
                    cursor.get_variance(1) + incremental_variance)
                self.n_buckets += 1
                cursor.compress_bucket_row(2)

                if next_node.n_buckets_in_row <= self.MAX_BUCKETS:
                    break
            else:
                break

            cursor = cursor.get_next_item()
            i += 1

    cdef bint _detect_change(self):
        """Detect concept change.

        This function is responsible for analysing different cutting points in
        the sliding window, to verify if there is a significant change.

        Returns
        -------
        bint
            If True then a change is detected.

        Notes
        -----
        If change was detected, one should verify the new window size,
        by reading the width property.

        """
        cdef:
            int i, k
            bint change_detected, exit_flag
            double n0, n1, n2, u0, u1, u2, v0, v1
            Item cursor

        change_detected = False
        exit_flag = False
        self.tick += 1
        n0 = 0
        if (self.tick % self.clock == 0) and (self.width > self.min_window_len):
            reduce_width = True
            while reduce_width:
                reduce_width = False
                exit_flag = False
                n0 = 0.0
                n1 = self.width
                u0 = 0.0
                u1 = self.total
                v0 = 0
                v1 = self.variance
                n2 = 0.0
                u2 = 0.0
                cursor = self.bucket_list.last
                i = self.last_bucket_idx

                while (not exit_flag) and (cursor is not None):
                    for k in range(cursor.n_buckets_in_row - 1):
                        n2 = self._bucket_size(i)
                        u2 = cursor.get_total(k)

                        if n0 > 0.0:
                            v0 += (
                                    cursor.get_variance(k) + n0 * n2
                                    * (u0 / n0 - u2 / n2) * (u0 / n0 - u2 / n2)
                                    / (n0 + n2)
                            )

                        if n1 > 0.0:
                            v1 -= (
                                    cursor.get_variance(k) + n1 * n2
                                    * (u1 / n1 - u2 / n2) * (u1 / n1 - u2 / n2)
                                    / (n1 + n2)
                            )

                        n0 += self._bucket_size(i)
                        n1 -= self._bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if (i == 0) and (k == cursor.n_buckets_in_row - 1):
                            exit_flag = True
                            break

                        abs_value = (u0 / n0) - (u1 / n1)

                        if (
                                n1 >= <double> self.mint_min_window_length
                                and n0 >= <double> self.mint_min_window_length
                                and self._cut_expression(n0, n1, abs_value, self.delta)
                        ):
                            self.detect = self.tick
                            if self.detect == 0:
                                self.detect = self.tick
                            elif self.detect_twice == 0:
                                self.detect_twice = self.tick

                            reduce_width = True
                            change_detected = True
                            if self.width > 0:
                                n0 -= self._delete_element()
                                exit_flag = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1

        self.total_width += self.width
        if change_detected:
            self.n_detections += 1

        return change_detected

    cdef bint _cut_expression(self, double n0, double n1,
                                double abs_value, double delta):
        cdef:
            double delta_prime, m, epsilon
        delta_prime = log(2 * log(self.width) / delta)
        m = ((1.0 / (n0 - self.mint_min_window_length + 1))
             + (1.0 / (n1 - self.mint_min_window_length + 1)))
        epsilon = sqrt(2 * m * self.variance_in_window * delta_prime) + 2 / 3 * delta_prime * m
        return fabs(abs_value) > epsilon


cdef class BucketList:
    """ A linked list for ADWIN' buckets.

    Stores ADWIN's bucket list composed of Item objects,
    where each element points to its predecessor and successor.

    """

    cdef:
        int count
        Item first, last
        # dict __dict__

    def __init__(self):
        self.count = 0
        self.first = None
        self.last = None

        self.add_to_head()

    def add_to_head(self):
        self.first = Item(self.first, None)
        if self.last is None:
            self.last = self.first

    def remove_from_head(self):
        self.first = self.first.get_next_item()
        if self.first is not None:
            self.first.set_previous(None)
        else:
            self.last = None
        self.count -= 1

    def add_to_tail(self):
        self.last = Item(None, self.last)
        if self.first is None:
            self.first = self.last
        self.count += 1

    def remove_from_tail(self):
        self.last = self.last.get_previous()
        if self.last is not None:
            self.last.set_next_item(None)
        else:
            self.first = None
        self.count -= 1

    @property
    def size(self):
        return self.count


cdef class Item:
    """Item to be used by the List object.

    The Item object, alongside the List object, are the two main data
    structures used for storing the relevant statistics for the ADWIN
    algorithm for change detection.

    Parameters
    ----------
    next_item
        Reference to the next Item in the List
    previous_item
        Reference to the previous Item in the List

    """
    # cdef dict __dict__
    cdef:
        Item next, previous
        int n_buckets_in_row, max_buckets
        np.ndarray bucket_total, bucket_variance

    def __init__(self, next_item=None, previous_item=None):
        self.next = next_item
        self.previous = previous_item
        if next_item is not None:
            next_item.previous = self
        if previous_item is not None:
            self.previous.next = self
        self.max_buckets = ADWINC.MAX_BUCKETS

        self.n_buckets_in_row = 0
        self.bucket_total = np.zeros(self.max_buckets + 1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets + 1, dtype=float)

    cdef void clear_buckets(self, int index):
        self.set_total(0, index)
        self.set_variance(0, index)

    cdef void insert_bucket(self, double value, double variance):
        cdef int new_item_idx = self.n_buckets_in_row
        self.n_buckets_in_row += 1
        self.set_total(value, new_item_idx)
        self.set_variance(variance, new_item_idx)

    def remove_bucket(self):
        self.compress_bucket_row(1)

    cdef void compress_bucket_row(self, int n_buckets):
        """ drop the front num_deleted buckets

        Parameters
        ----------
        n_buckets
            The number of buckets to be cleared.
        Returns
        -------

        """
        cdef int i
        for i in range(n_buckets, ADWINC.MAX_BUCKETS + 1):
            self.bucket_total[i - n_buckets] = self.bucket_total[i]
            self.bucket_variance[i - n_buckets] = self.bucket_variance[i]

        for i in range(1, n_buckets + 1):
            self.clear_buckets(ADWINC.MAX_BUCKETS - i + 1)

        self.n_buckets_in_row -= n_buckets

    cdef Item get_next_item(self):
        return self.next

    cdef void set_next_item(self, Item next_item):
        self.next = next_item

    cdef Item get_previous(self):
        return self.previous

    cdef void set_previous(self, Item previous):
        self.previous = previous

    cdef double get_total(self, int index):
        return self.bucket_total[index]

    cdef double get_variance(self, int index):
        return self.bucket_variance[index]

    cdef void set_total(self, double value, int index):
        self.bucket_total[index] = value

    cdef void set_variance(self, double value, int index):
        self.bucket_variance[index] = value
