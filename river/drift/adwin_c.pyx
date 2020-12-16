# cython: boundscheck=False

from libc.math cimport sqrt, log, fabs, pow
import numpy as np
cimport numpy as np
from collections import deque


cdef class ADWINC:
    """ The Cython helper class for ADWIN

    Parameters
    ----------
    delta
        Confidence value.

    """
    cdef:
        # dict __dict__
        List list_row_bucket
        double delta, _total, _variance, _v, mdbl_delta, mdbl_width, _width
        int last_bucket_row, bucket_number, mint_min_window_longitude, mint_min_window_length, mint_time, \
            _n_detections, detect_twice, mint_clock, bucket_num_max, detect
        bint _in_concept_change

    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_row = 0
        self.list_row_bucket = None
        self._total = 0
        self._variance = 0
        self._v = 0.0
        self._width = 0
        self.bucket_number = 0

        self.__init_buckets()

        # other default values
        self.mint_min_window_longitude = 10

        self.mdbl_delta = .002
        self.mint_time = 0
        self.mdbl_width = 0

        self.detect = 0
        self._n_detections = 0
        self.detect_twice = 0
        self.mint_clock = 32

        self.bucket_num_max = 0
        self.mint_min_window_length = 5

    def reset(self):
        """Reset the change detector.
        """
        self.__init__(delta=self.delta)

    @property
    def delta(self):
        return self.delta

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
        return self._v

    @property
    def estimation(self):
        """Error estimation"""
        if self._width == 0:
            return 0
        return self._total / self._width

    cdef void __init_buckets(self):
        """ Initialize the bucket's List and statistics

        Set all statistics to 0 and create a new bucket List.

        """
        self.list_row_bucket = List()
        self.last_bucket_row = 0
        self._total = 0
        self._variance = 0
        self._v = 0.0
        self._width = 0
        self.bucket_number = 0

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
            Indicate if a drift or a warning is
            detected.

        """
        cdef double _w = self._width

        self._width += 1

        self.__insert_element_bucket(0, value, self.list_row_bucket.first())

        if self._width > 1:
            self._variance += _w * (value - self._total / _w) * (value - self._total / _w) / self._width
            self._v = self._variance / self._width

        self._total += value
        self.__compress_buckets()

        return self._detect_change()

    cdef void __insert_element_bucket(self, double variance, double value, Item node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

    # @staticmethod
    cdef double _bucket_size(self, int row):
        return pow(2, row)

    cdef double _delete_element(self):
        """Delete an item from the bucket list.

        Deletes the last item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        The bucket size from the updated bucket

        """
        cdef:
            double n1, u1
            Item node
            double incremental_variance

        node = self.list_row_bucket.last()
        n1 = self._bucket_size(self.last_bucket_row)
        self._width -= n1
        self._total -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = node.get_variance(0) + n1 * self._width * (
            u1 - self._total / self._width) * (u1 - self._total / self._width) / (
            n1 + self._width)
        self._variance -= incremental_variance
        self._v = self._variance / self._width
        node.remove_bucket()
        self.bucket_number -= 1

        if node.bucket_size_row == 0:
            self.list_row_bucket.remove_from_tail()
            self.last_bucket_row -= 1

        return n1

    cdef void __compress_buckets(self):

        cdef:
            int i, k
            double n1, n2, u1, u2, incremental_variance
            Item cursor, next_node

        cursor = self.list_row_bucket.first()
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
                    cursor.get_variance(1) + incremental_variance)
                self.bucket_number += 1
                cursor.compress_bucket_row(2)

                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break

            cursor = cursor.get_next_item()
            i += 1

    cdef bint _detect_change(self):
        """Detects concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from
        Time-Changing Data with Adaptive Windowing'. The general idea is to keep
        statistics from a window of variable size while detecting concept drift.

        This function is responsible for analysing different cutting points in
        the sliding window, to verify if there is a significant change in concept.

        Returns
        -------
        bint
            Indicate if a drift or a warning is
            detected.

        Notes
        -----
        If change was detected, one should verify the new window size, by reading the width
        property.

        """
        cdef:
            bint bln_change, bln_exit, bln_bucket_deleted
            double n0, n1, n2, u0, u1, u2
            Item cursor

        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (
                self._width > self.mint_min_window_longitude):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self._width
                u0 = 0
                u1 = self._total
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket.last()
                i = self.last_bucket_row

                while (not bln_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row - 1):
                        if (i == 0) and (k == cursor.bucket_size_row - 1):
                            bln_exit = True
                            break

                        n2 = self._bucket_size(i)
                        u2 = cursor.get_total(k)

                        n0 += self._bucket_size(i)
                        n1 -= self._bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)
                        abs_value = 1. * ((u0 / n0) - (u1 / n1))
                        if (n1 >= self.mint_min_window_length) \
                                and (n0 >= self.mint_min_window_length) \
                                and (
                                self.__bln_cut_expression(n0, n1, u0, u1, abs_value,
                                                          self.delta)):
                            bln_bucket_deleted = True  # noqa: F841
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            if self._width > 0:
                                n0 -= self._delete_element()
                                bln_exit = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1
        self.mdbl_width += self._width
        if bln_change:
            self._n_detections += 1
        self._in_concept_change = bln_change

        return self._in_concept_change

    cdef double __bln_cut_expression(self, double n0, double n1, double u0, double u1,
                                     double abs_value, double delta):
        cdef:
            double n, dd, v, m, epsilon

        n = self._width
        dd = log(2 * log(n) / delta)
        v = self._v
        m = (1. / (n0 - self.mint_min_window_length + 1)) + \
            (1. / (n1 - self.mint_min_window_length + 1))
        epsilon = sqrt(2 * m * v * dd) + 1. * 2 / 3 * dd * m
        return fabs(abs_value) > epsilon


cdef class List:
    """ A linked list object for ADWIN algorithm.

    Used for storing ADWIN's bucket list. Is composed of Item objects.
    Acts as a linked list, where each element points to its predecessor
    and successor.

    """

    cdef:
        int _count
        dict __dict__

    def __init__(self):
        self._nodes = deque()
        self.add_to_head()
        self._count = 0

    cdef Item first(self):
        if self._nodes and len(self._nodes) > 0:
            return self._nodes[0]
        return None

    cdef Item last(self):
        if self._nodes and len(self._nodes) > 0:
            return self._nodes[-1]
        return None

    cdef void add_to_head(self):
        cdef Item head, new_head

        head = self.first()
        new_head = Item(head, None)
        self._nodes.appendleft(new_head)
        if head:
            self.first().set_previous(new_head)
        self._count += 1

    cdef void remove_from_head(self):
        self._nodes.popleft()
        if self.first():
            self.first().set_previous(None)
        self._count -= 1

    cdef void add_to_tail(self):
        cdef Item tail, new_tail

        tail = self.last()
        new_tail = Item(None, tail)
        if tail:
            tail.next = new_tail
        self._nodes.append(new_tail)
        self._count += 1

    cdef void remove_from_tail(self):
        self._nodes.pop()
        if self.last():
            self.last().set_next_item(None)
        self._count -= 1

    cdef int size(self):
        return self._count


cdef class Item:
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
    # cdef dict __dict__
    cdef:
        Item next, previous
        int bucket_size_row, max_buckets
        np.ndarray bucket_total, bucket_variance

    def __init__(self, next_item=None, previous_item=None):
        self.next = next_item
        self.previous = previous_item
        if next_item is not None:
            next_item.previous = self
        if previous_item is not None:
            self.previous.next = self
        self.max_buckets = ADWINC.MAX_BUCKETS
        self.reset()

    cdef void __clear_buckets(self, int start_index):
        """ Reset the algorithm's statistics and window

        Parameters
        ----------
        start_index: start position
            The start position to clear
        """
        self.bucket_total[start_index::] = 0.0
        self.bucket_variance[start_index::] = 0.0

    cdef void reset(self):
        """ Reset the algorithm's statistics and window
        """
        self.bucket_size_row = 0
        self.bucket_total = np.zeros(self.max_buckets + 1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets + 1, dtype=float)

    # def __clear_buckets(self, index):
    #     self.set_total(0, index)
    #     self.set_variance(0, index)

    cdef void insert_bucket(self, double value, double variance):
        cdef int new_item = self.bucket_size_row
        self.bucket_size_row += 1
        self.set_total(value, new_item)
        self.set_variance(variance, new_item)

    def remove_bucket(self):
        self.compress_bucket_row(1)

    cdef void compress_bucket_row(self, num_deleted=1):
        """ drop the front num_deleted buckets

        Parameters
        ----------
        num_deleted: int
            The number of buckets to be cleared.
        Returns
        -------

        """
        for i in range(num_deleted, ADWINC.MAX_BUCKETS + 1):
            self.bucket_total[i - num_deleted] = self.bucket_total[i]
            self.bucket_variance[i - num_deleted] = self.bucket_variance[i]

        self.bucket_size_row -= num_deleted
        self.bucket_total[self.bucket_size_row::] = 0.0
        self.bucket_variance[self.bucket_size_row::] = 0.0

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
