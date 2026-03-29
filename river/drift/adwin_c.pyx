# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport fabs, log, sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memmove


def _rebuild_bucket(int max_size, int current_idx, list totals, list variances):
    """Reconstruct a Bucket from pickled state."""
    cdef Bucket b = Bucket(max_size)
    b.current_idx = current_idx
    cdef int i
    for i in range(current_idx):
        b.total_array[i] = totals[i]
        b.variance_array[i] = variances[i]
    return b


cdef class Bucket:
    """A bucket class to keep statistics using C arrays for speed.

    A bucket stores the summary structure for a contiguous set of data elements.
    In this implementation fixed-size C arrays are used for efficiency. The index
    of the "current" element is used to simulate the dynamic size of the bucket.

    """
    cdef:
        int current_idx, max_size
        double *total_array
        double *variance_array

    def __cinit__(self, int max_size):
        self.max_size = max_size
        self.current_idx = 0
        cdef int alloc_size = max_size + 1
        self.total_array = <double *>malloc(alloc_size * sizeof(double))
        self.variance_array = <double *>malloc(alloc_size * sizeof(double))
        if self.total_array == NULL or self.variance_array == NULL:
            raise MemoryError("Failed to allocate Bucket arrays")
        memset(self.total_array, 0, alloc_size * sizeof(double))
        memset(self.variance_array, 0, alloc_size * sizeof(double))

    def __reduce__(self):
        # Extract array data as Python lists for pickling
        cdef list totals = []
        cdef list variances = []
        cdef int i
        for i in range(self.current_idx):
            totals.append(self.total_array[i])
            variances.append(self.variance_array[i])
        return (_rebuild_bucket, (self.max_size, self.current_idx, totals, variances))

    def __dealloc__(self):
        if self.total_array != NULL:
            free(self.total_array)
            self.total_array = NULL
        if self.variance_array != NULL:
            free(self.variance_array)
            self.variance_array = NULL

    cdef inline void insert_data(self, double value, double variance) noexcept nogil:
        self.total_array[self.current_idx] = value
        self.variance_array[self.current_idx] = variance
        self.current_idx += 1

    cdef inline void remove(self) noexcept nogil:
        self.compress(1)

    cdef void compress(self, int n_elements) noexcept nogil:
        cdef int remaining = self.current_idx - n_elements
        if remaining > 0:
            memmove(self.total_array, self.total_array + n_elements, remaining * sizeof(double))
            memmove(self.variance_array, self.variance_array + n_elements, remaining * sizeof(double))
        # Zero out the vacated slots
        cdef int i
        for i in range(remaining, self.current_idx):
            self.total_array[i] = 0.0
            self.variance_array[i] = 0.0
        self.current_idx -= n_elements

    cdef inline double get_total_at(self, int index) noexcept nogil:
        return self.total_array[index]

    cdef inline double get_variance_at(self, int index) noexcept nogil:
        return self.variance_array[index]


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
        double delta, total, variance, total_width, width
        int n_buckets, grace_period, min_window_length, tick, n_detections,\
            clock, max_n_buckets, detect, detect_twice, max_buckets
        list _bucket_list

    def __init__(self, delta=.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10):
        self.delta = delta
        self._bucket_list = [Bucket(max_size=max_buckets)]
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
        cdef Bucket bucket = <Bucket>self._bucket_list[0]
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

    cdef double _delete_element(self):
        cdef int deque_len = len(self._bucket_list)
        cdef Bucket bucket = <Bucket>self._bucket_list[deque_len - 1]
        cdef double n = <double>(1 << (deque_len - 1))  # 2^(deque_len-1)
        cdef double u = bucket.get_total_at(0)     # total of bucket
        cdef double mu = u / n                   # mean of bucket
        cdef double v = bucket.get_variance_at(0)  # variance of bucket

        # Update width, total and variance
        self.width -= n
        self.total -= u
        cdef double mu_window = self.total / self.width     # mean of the window
        cdef double incremental_variance = (
                v + n * self.width * (mu - mu_window) * (mu - mu_window)
                / (n + self.width)
        )
        self.variance -= incremental_variance

        bucket.remove()
        self.n_buckets -= 1

        if bucket.current_idx == 0:
            self._bucket_list.pop()

        return n

    cdef void _compress_buckets(self):

        cdef:
            unsigned int idx, k
            int deque_len
            double n1, mu1, mu2, temp, total12, v12
            Bucket bucket, next_bucket

        bucket = <Bucket>self._bucket_list[0]
        idx = 0
        while bucket is not None:
            k = bucket.current_idx
            # Merge buckets if there are more than max_buckets
            if k == self.max_buckets + 1:
                deque_len = len(self._bucket_list)
                if idx + 1 < deque_len:
                    next_bucket = <Bucket>self._bucket_list[idx + 1]
                else:
                    next_bucket = Bucket(max_size=self.max_buckets)
                    self._bucket_list.append(next_bucket)
                n1 = <double>(1 << idx)              # length of bucket: 2^idx
                mu1 = bucket.get_total_at(0) / n1    # mean of bucket 1
                mu2 = bucket.get_total_at(1) / n1    # mean of bucket 2

                # Combine total and variance of adjacent buckets
                total12 = bucket.get_total_at(0) + bucket.get_total_at(1)
                # n1 * n2 / (n1 + n2) = n1 / 2 since n1 == n2
                temp = n1 * 0.5 * (mu1 - mu2) * (mu1 - mu2)
                v12 = bucket.get_variance_at(0) + bucket.get_variance_at(1) + temp
                next_bucket.insert_data(total12, v12)
                self.n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.max_buckets:
                    break
            else:
                break

            deque_len = len(self._bucket_list)
            if idx + 1 < deque_len:
                bucket = <Bucket>self._bucket_list[idx + 1]
            else:
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

        Babcock, B., Datar, M., Motwani, R., & O'Callaghan, L. (2003).
        Maintaining Variance and k-Medians over Data Stream Windows.
        Proceedings of the ACM SIGACT-SIGMOD-SIGART
        Symposium on Principles of Database Systems, 22, 234–243.
        https://doi.org/10.1145/773153.773176

        """
        cdef:
            int idx, deque_len
            unsigned int k
            bint change_detected, exit_flag, reduce_width
            double n0, n1, n2, u0, u1, u2, v0, v1
            double mu0, mu1, mu2, delta_mean, bucket_size
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
                v0 = 0.0            # variance of window 0
                v1 = self.variance  # variance of window 1

                deque_len = len(self._bucket_list)

                # Evaluate each window cut (W_0, W_1)
                for idx in range(deque_len - 1, -1, -1):
                    if exit_flag:
                        break
                    bucket = <Bucket>self._bucket_list[idx]
                    bucket_size = <double>(1 << idx)

                    for k in range(bucket.current_idx):
                        n2 = bucket_size   # length of window 2
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
                        n0 += bucket_size
                        n1 -= bucket_size
                        u0 += bucket.get_total_at(k)
                        u1 -= bucket.get_total_at(k)

                        if (idx == 0) and (k == <unsigned int>(bucket.current_idx - 1)):
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
        # Inline variance_in_window to avoid Python property dispatch
        epsilon = (sqrt(2 * m_recip * (self.variance / self.width) * delta_prime)
                   + 2.0 / 3.0 * delta_prime * m_recip)

        return fabs(delta_mean) > epsilon
