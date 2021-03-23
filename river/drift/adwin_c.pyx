# cython: boundscheck=False

from libc.math cimport sqrt, log, fabs, pow
import numpy as np
cimport numpy as np
from typing import Deque
from collections import deque


cdef class ADWINC:
    """ The helper class for ADWIN

    Parameters
    ----------
    delta
        Confidence value.

    """
    cdef:
        dict __dict__
        double delta, total, variance, total_width, width
        int n_buckets, min_window_len,\
            mint_min_window_length, tick, n_detections,\
            clock, max_n_buckets, detect, detect_twice

    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        # default values affected by init_bucket()
        self.delta = delta
        self.bucket_deque = None
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
        self.bucket_deque: Deque['Bucket'] = deque([Bucket()])
        self.total = 0.0
        self.variance = 0.0
        self.width = 0.0
        self.n_buckets = 0

    def update(self, value: float):
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
        bool
            If True then a change is detected.

        """
        return self._update(value)

    cdef bint _update(self, double value):

        cdef double incremental_variance

        self.width += 1

        self._insert_element(value, 0.0)
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

    cdef void _insert_element(self, double value, double variance):
        cdef Bucket bucket = self.bucket_deque[0]
        bucket.insert_data(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

    cdef double _bucket_size(self, int row):
        return pow(2, row)

    cdef double _delete_element(self):
        """Delete an item from the bucket deque.

        Deletes the last item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        The size of the item being dropped

        """
        cdef:
            double n1, u1
            Bucket bucket
            double incremental_variance

        bucket = self.bucket_deque[-1]
        n1 = self._bucket_size(len(self.bucket_deque) - 1)
        self.width -= n1
        self.total -= bucket.get_total_at(0)
        u1 = bucket.get_total_at(0) / n1
        incremental_variance = (
                bucket.get_variance_at(0) + n1 * self.width
                * (u1 - self.total / self.width) * (u1 - self.total / self.width)
                / (n1 + self.width)
        )
        self.variance -= incremental_variance
        bucket.remove()
        self.n_buckets -= 1

        if bucket.current_idx == 0:
            self.bucket_deque.pop()

        return n1

    cdef void _compress_buckets(self):

        cdef:
            int idx, k
            double n1, n2, u1, u2, incremental_variance
            Bucket bucket, next_bucket

        for idx, bucket in enumerate(self.bucket_deque):
            k = bucket.current_idx
            # If the row is full, merge buckets
            if k == self.MAX_BUCKETS + 1:
                try:
                    next_bucket = self.bucket_deque[idx + 1]
                except IndexError:
                    self.bucket_deque.append(Bucket())
                    next_bucket = self.bucket_deque[-1]
                n1 = self._bucket_size(idx)
                n2 = self._bucket_size(idx)
                u1 = bucket.get_total_at(0) / n1
                u2 = bucket.get_total_at(1) / n2
                incremental_variance = n1 * n2 * ((u1 - u2) * (u1 - u2)) / (n1 + n2)
                next_bucket.insert_data(
                    bucket.get_total_at(0) + bucket.get_total_at(1),
                    bucket.get_variance_at(1) + incremental_variance
                )
                self.n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.MAX_BUCKETS:
                    break
            else:
                break


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
            int idx, k
            bint change_detected, exit_flag
            double n0, n1, n2, u0, u1, u2, v0, v1
            Bucket bucket

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

                for idx in range(len(self.bucket_deque) - 1, -1 , -1):
                    if exit_flag:
                        break
                    bucket = self.bucket_deque[idx]

                    for k in range(bucket.current_idx - 1):
                        n2 = self._bucket_size(idx)
                        u2 = bucket.get_total_at(k)

                        if n0 > 0.0:
                            v0 += (
                                    bucket.get_variance_at(k) + n0 * n2
                                    * (u0 / n0 - u2 / n2) * (u0 / n0 - u2 / n2)
                                    / (n0 + n2)
                            )

                        if n1 > 0.0:
                            v1 -= (
                                    bucket.get_variance_at(k) + n1 * n2
                                    * (u1 / n1 - u2 / n2) * (u1 / n1 - u2 / n2)
                                    / (n1 + n2)
                            )

                        n0 += self._bucket_size(idx)
                        n1 -= self._bucket_size(idx)
                        u0 += bucket.get_total_at(k)
                        u1 -= bucket.get_total_at(k)

                        if (idx == 0) and (k == bucket.current_idx - 1):
                            exit_flag = True
                            break

                        abs_value = (u0 / n0) - (u1 / n1)

                        if (
                                n1 >= <double> self.mint_min_window_length
                                and n0 >= <double> self.mint_min_window_length
                                and self._evaluate_cut(n0, n1, abs_value, self.delta)
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

        self.total_width += self.width
        if change_detected:
            self.n_detections += 1

        return change_detected

    cdef bint _evaluate_cut(self, double n0, double n1,
                            double abs_value, double delta):
        cdef:
            double delta_prime, m, epsilon
        delta_prime = log(2 * log(self.width) / delta)
        m = ((1.0 / (n0 - self.mint_min_window_length + 1))
             + (1.0 / (n1 - self.mint_min_window_length + 1)))
        epsilon = sqrt(2 * m * self.variance_in_window * delta_prime) + 2 / 3 * delta_prime * m
        return fabs(abs_value) > epsilon


cdef class Bucket:
    """ A bucket class to keep statistics.

    In this implementation fixed-size arrays are used for efficiency. The index
    of the "current" item is used to simulate the dynamic size of the row.

    """
    cdef:
        int current_idx, max_size
        np.ndarray total_array, variance_array

    def __init__(self):
        self.max_size = ADWINC.MAX_BUCKETS

        self.current_idx = 0
        self.total_array = np.zeros(self.max_size + 1, dtype=float)
        self.variance_array = np.zeros(self.max_size + 1, dtype=float)

    cdef void clear_at(self, int index):
        self.set_total_at(0.0, index)
        self.set_variance_at(0.0, index)

    cdef void insert_data(self, double value, double variance):
        self.set_total_at(value, self.current_idx)
        self.set_variance_at(variance, self.current_idx)
        self.current_idx += 1

    cdef void remove(self):
        self.compress(1)

    cdef void compress(self, int n_items):
        """ Drop the front n_items

        Parameters
        ----------
        n_items
            The number of buckets to be cleared.

        """
        cdef int i
        for i in range(n_items, ADWINC.MAX_BUCKETS + 1):
            self.total_array[i - n_items] = self.total_array[i]
            self.variance_array[i - n_items] = self.variance_array[i]

        for i in range(1, n_items + 1):
            self.clear_at(ADWINC.MAX_BUCKETS - i + 1)

        self.current_idx -= n_items

    cdef double get_total_at(self, int index):
        return self.total_array[index]

    cdef double get_variance_at(self, int index):
        return self.variance_array[index]

    cdef void set_total_at(self, double value, int index):
        self.total_array[index] = value

    cdef void set_variance_at(self, double value, int index):
        self.variance_array[index] = value
