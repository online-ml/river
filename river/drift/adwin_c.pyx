# cython: boundscheck=False

from libc.math cimport fabs, log, pow, sqrt

import numpy as np

cimport numpy as np

from collections import deque
from typing import Deque


cdef class AdaptiveWindowing:
    """ The helper class for ADWIN

    Parameters
    ----------
    delta
        Confidence value.
    clock
        How often ADWIN should check for change. 1 means every new data point, default is 32. Higher
         values speed up processing, but may also lead to increased delay in change detection.
    max_buckets
        The maximum number of buckets of each size that ADWIN should keep before merging buckets
        (default is 5).
    min_window_length
        The minimum length of each subwindow (default is 5). Lower values may decrease delay in
        change detection but may also lead to more false positives.
    grace_period
        ADWIN does not perform any change detection until at least this many data points have
        arrived (default is 10).

    """
    cdef:
        dict __dict__
        double delta, total, variance, total_width, width
        int n_buckets, grace_period, min_window_length, tick, n_detections,\
            clock, max_n_buckets, detect, detect_twice, max_buckets

    def __init__(self, delta=.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10):
        self.delta = delta
        self.bucket_deque: Deque['Bucket'] = deque([Bucket(max_size=max_buckets)])
        self.total = 0.
        self.variance = 0.
        self.width = 0.
        self.n_buckets = 0
        self.grace_period = grace_period
        self.tick = 0
        self.total_width = 0
        self.n_detections = 0
        self.clock = clock
        self.max_n_buckets = 0
        self.min_window_length = min_window_length
        self.max_buckets = max_buckets

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
        # Increment window with one element
        self._insert_element(value, 0.0)

        return self._detect_change()

    cdef void _insert_element(self, double value, double variance):
        cdef Bucket bucket = self.bucket_deque[0]
        bucket.insert_data(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

        # Update width, variance and total
        self.width += 1
        cdef double incremental_variance = 0.0
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

    @staticmethod
    def _calculate_bucket_size(row: int):
        return pow(2, row)

    cdef double _delete_element(self):
        cdef Bucket bucket = self.bucket_deque[-1]
        cdef double n = self._calculate_bucket_size(len(self.bucket_deque) - 1) # length of bucket
        cdef double u = bucket.get_total_at(0)     # total of bucket
        cdef double mu = u / n                   # mean of bucket
        cdef double v = bucket.get_variance_at(0)  # variance of bucket

        # Update width, total and variance
        self.width -= n
        self.total -= u
        mu_window = self.total / self.width     # mean of the window
        cdef double incremental_variance = (
                v + n * self.width * (mu - mu_window) * (mu - mu_window)
                / (n + self.width)
        )
        self.variance -= incremental_variance

        bucket.remove()
        self.n_buckets -= 1

        if bucket.current_idx == 0:
            self.bucket_deque.pop()

        return n

    cdef void _compress_buckets(self):

        cdef:
            unsigned int idx, k
            double n1, n2, mu1, mu2, temp, total12
            Bucket bucket, next_bucket

        bucket = self.bucket_deque[0]
        idx = 0
        while bucket is not None:
            k = bucket.current_idx
            # Merge buckets if there are more than max_buckets
            if k == self.max_buckets + 1:
                try:
                    next_bucket = self.bucket_deque[idx + 1]
                except IndexError:
                    self.bucket_deque.append(Bucket(max_size=self.max_buckets))
                    next_bucket = self.bucket_deque[-1]
                n1 = self._calculate_bucket_size(idx)   # length of bucket 1
                n2 = self._calculate_bucket_size(idx)   # length of bucket 2
                mu1 = bucket.get_total_at(0) / n1       # mean of bucket 1
                mu2 = bucket.get_total_at(1) / n2       # mean of bucket 2

                # Combine total and variance of adjacent buckets
                total12 = bucket.get_total_at(0) + bucket.get_total_at(1)
                temp = n1 * n2 * (mu1 - mu2) * (mu1 - mu2) / (n1 + n2)
                v12 = bucket.get_variance_at(0) + bucket.get_variance_at(1) + temp
                next_bucket.insert_data(total12, v12)
                self.n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.max_buckets:
                    break
            else:
                break

            try:
                bucket = self.bucket_deque[idx + 1]
            except IndexError:
                bucket = None
            idx += 1

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
        Variance calculation is based on:

        Babcock, B., Datar, M., Motwani, R., & O’Callaghan, L. (2003).
        Maintaining Variance and k-Medians over Data Stream Windows.
        Proceedings of the ACM SIGACT-SIGMOD-SIGART
        Symposium on Principles of Database Systems, 22, 234–243.
        https://doi.org/10.1145/773153.773176

        """
        cdef:
            unsigned int idx, k
            bint change_detected, exit_flag
            double n0, n1, n2, u0, u1, u2, v0, v1
            Bucket bucket

        change_detected = False
        exit_flag = False
        self.tick += 1

        # Reduce window
        if (self.tick % self.clock == 0) and (self.width > self.grace_period):
            reduce_width = True
            while reduce_width:
                reduce_width = False
                exit_flag = False
                n0 = 0.0            # length of window 0
                n1 = self.width     # length of window 1
                u0 = 0.0            # total of window 0
                u1 = self.total     # total of window 1
                v0 = 0              # variance of window 0
                v1 = self.variance  # variance of window 1

                # Evaluate each window cut (W_0, W_1)
                for idx in range(len(self.bucket_deque) - 1, -1 , -1):
                    if exit_flag:
                        break
                    bucket = self.bucket_deque[idx]

                    for k in range(bucket.current_idx - 1):
                        n2 = self._calculate_bucket_size(idx)   # length of window 2
                        u2 = bucket.get_total_at(k)             # total of window 2
                        # Warning: means are calculated inside the loop to get updated values.
                        mu2 = u2 / n2   # mean of window 2

                        if n0 > 0.0:
                            mu0 = u0 / n0  # mean of window 0
                            v0 += (
                                    bucket.get_variance_at(k) + n0 * n2
                                    * (mu0 - mu2) * (mu0 - mu2)
                                    / (n0 + n2)
                            )

                        if n1 > 0.0:
                            mu1 = u1 / n1  # mean of window 1
                            v1 -= (
                                    bucket.get_variance_at(k) + n1 * n2
                                    * (mu1 - mu2) * (mu1 - mu2)
                                    / (n1 + n2)
                            )

                        # Update window 0 and 1
                        n0 += self._calculate_bucket_size(idx)
                        n1 -= self._calculate_bucket_size(idx)
                        u0 += bucket.get_total_at(k)
                        u1 -= bucket.get_total_at(k)

                        if (idx == 0) and (k == bucket.current_idx - 1):
                            exit_flag = True    # We are done
                            break

                        # Check if delta_mean < epsilon_cut holds
                        # Note: Must re-calculate means per updated values
                        delta_mean = (u0 / n0) - (u1 / n1)
                        if (
                                n1 >= self.min_window_length
                                and n0 >= self.min_window_length
                                and self._evaluate_cut(n0, n1, delta_mean, self.delta)
                        ):
                            # Change detected

                            reduce_width = True
                            change_detected = True
                            if self.width > 0:
                                # Reduce the width of the window
                                n0 -= self._delete_element()
                                exit_flag = True    # We are done
                                break

        self.total_width += self.width
        if change_detected:
            self.n_detections += 1

        return change_detected

    cdef bint _evaluate_cut(self, double n0, double n1,
                            double delta_mean, double delta):
        cdef:
            double delta_prime, m_recip, epsilon
        delta_prime = log(2 * log(self.width) / delta)
        # Use reciprocal of m to avoid extra divisions when calculating epsilon
        m_recip = ((1.0 / (n0 - self.min_window_length + 1))
                   + (1.0 / (n1 - self.min_window_length + 1)))
        epsilon = (sqrt(2 * m_recip * self.variance_in_window * delta_prime)
                   + 2 / 3 * delta_prime * m_recip)
        return fabs(delta_mean) > epsilon


cdef class Bucket:
    """ A bucket class to keep statistics.

    A bucket stores the summary structure for a contiguous set of data elements.
    In this implementation fixed-size arrays are used for efficiency. The index
    of the "current" element is used to simulate the dynamic size of the bucket.

    """
    cdef:
        int current_idx, max_size
        np.ndarray total_array, variance_array

    def __init__(self, max_size):
        self.max_size = max_size

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

    cdef void compress(self, int n_elements):
        cdef unsigned int i
        cdef int window_len = len(self.total_array)
        # Remove first n_elements by shifting elements to the left
        for i in range(n_elements, window_len):
            self.total_array[i - n_elements] = self.total_array[i]
            self.variance_array[i - n_elements] = self.variance_array[i]
        # Clear remaining elements
        for i in range(window_len - n_elements, window_len):
            self.clear_at(i)

        self.current_idx -= n_elements

    cdef double get_total_at(self, int index):
        return self.total_array[index]

    cdef double get_variance_at(self, int index):
        return self.variance_array[index]

    cdef void set_total_at(self, double value, int index):
        self.total_array[index] = value

    cdef void set_variance_at(self, double value, int index):
        self.variance_array[index] = value
