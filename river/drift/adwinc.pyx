# cython: boundscheck=False

from libc.math cimport sqrt, log, fabs, pow
from sys import getsizeof
from collections import deque

from river.base import DriftDetector

cdef class AdWinList:
    """ A linked list object for ADWIN algorithm.

    Used for storing ADWIN's bucket list. Is composed of Item objects.
    Acts as a linked list, where each element points to its predecessor
    and successor.

    """
    cdef dict __dict__
    def __init__(self, max_buckets):
        self._nodes = deque()
        self._count = 0
        self._max_buckets = max_buckets
        self.add_head()

    cpdef double memory_size(self):
        cdef:
            double size
            AdWinListNode node

        size = getsizeof(self)
        size += getsizeof(self._nodes)
        size += getsizeof(self._count)
        size += getsizeof(self._max_buckets)
        size += sum([node.memory_size() for node in self._nodes])
        return size

    @property
    def head(self) -> AdWinListNode:
        if not self._nodes:
            return None
        return self._nodes[0]

    @property
    def tail(self) -> AdWinListNode:
        if not self._nodes:
            return None
        return self._nodes[-1]

    def __getitem__(self, int item):
        if item < len(self._nodes):
            return self._nodes[item]
        return None

    def __iter__(self):
        return enumerate(self._nodes)

    cpdef add_head(self):
        cdef AdWinListNode new_node

        new_node = AdWinListNode(self._max_buckets)
        new_node._prev = None
        new_node._next = self.head
        if self.head:
            self.head._prev = new_node
        self._nodes.appendleft(new_node)
        self._count += 1

    cpdef remove_head(self):
        cdef AdWinListNode head

        self._nodes.popleft()
        head = self.head
        head._prev = None
        self._count -= 1

    cpdef add_tail(self):
        cdef AdWinListNode new_node, tail

        new_node = AdWinListNode(self._max_buckets)
        new_node._prev = self.tail
        new_node._next = None
        tail = self.tail
        if tail:
            tail._next = new_node
        self._nodes.append(new_node)
        self._count += 1

    cpdef remove_tail(self):
        cdef AdWinListNode tail

        self._nodes.pop()
        tail = self.tail
        tail._next = None
        self._count -= 1


cdef class AdWinListNode:
    """ Item to be used by the List object.

    The Item object, alongside the List object, are the two main data
    structures used for storing the relevant statistics for the ADWIN
    algorithm for change detection.

    """

    cdef dict __dict__
    def __init__(self, int max_buckets):
        self._max_buckets = max_buckets
        self._size = 0
        self._next = None
        self._prev = None
        self._sum = []
        self._variance = []
        for i in range(self._max_buckets + 1):
            self._sum.append(0.0)
            self._variance.append(0.0)

    cpdef double memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._max_buckets)
        size += getsizeof(self._size)
        size += getsizeof(self._next)
        size += getsizeof(self._prev)
        size += getsizeof(self._sum)
        size += getsizeof(self._variance)
        return size

    cpdef add_back(self, double value, double var):
        self._sum[self._size] = value
        self._variance[self._size] = var
        self._size += 1

    cpdef drop_front(self, int n=1):
        cdef int k

        for k in range(n, self._max_buckets + 1):
            self._sum[k-n] = self._sum[k]
            self._variance[k - n] = self._variance[k]
        for k in range(1, n + 1):
            self._sum[self._max_buckets - k + 1] = 0.0
            self._variance[self._max_buckets - k + 1] = 0.0
        self._size -= n


cdef class ADWINC:
    cdef dict __dict__
    def __init__(self, delta, mint_clock=32, min_long_wind=10, MAXBUCKETS=5):
        self._delta = delta
        self._MAXBUCKETS = MAXBUCKETS
        self._buckets = AdWinList(self._MAXBUCKETS)
        self._min_long_wind = min_long_wind
        self._mint_time = 0.0
        self._mint_clock = mint_clock
        self._mdbl_error = 0.0
        self._mdbl_width = 0.0
        self._last_bucket = 0
        self._sum = 0.0
        self._W = 0.0
        self._var = 0.0
        self._bucket_n = 0

    cpdef double memory_size(self):
        cdef double size = getsizeof(self)
        size += getsizeof(self._delta)
        size += getsizeof(self._MAXBUCKETS)
        size += getsizeof(self._buckets)
        size += getsizeof(self._min_long_wind)
        size += getsizeof(self._mint_time)
        size += getsizeof(self._mint_clock)
        size += getsizeof(self._mdbl_error)
        size += getsizeof(self._mdbl_width)
        size += getsizeof(self._last_bucket)
        size += getsizeof(self._sum)
        size += getsizeof(self._W)
        size += getsizeof(self._var)
        size += getsizeof(self._bucket_n)
        size += self._buckets.memory_size()
        return size

    cpdef double get_estimation(self):
        if self._W > 0:
            return self._sum / self._W
        else:
            return 0

    cpdef update(self, double value):
        self.insert_element(value)
        self.compress_buckets()
        return self.drop_check_drift()

    cpdef print_info(self):
        cdef int i, k
        cdef AdWinListNode it

        it = self._buckets.tail
        if it is None:
            print("empty")

        i = self._last_bucket
        while True:
            for k in range(it._size-1, -1, -1):
                print(f'{i} [{it._sum[k]} / {self.bucket_size(i)}]')
            print()
            it = it._prev
            i -= 1
            if it is None:
                break

    @property
    def length(self) -> int:
        return self._W

    cpdef insert_element(self, double value):
        self._W += 1
        self._buckets.head.add_back(float(value), 0.0)
        self._bucket_n += 1

        if self._W > 1:
            self._var += ((self._W - 1) * (value - self._sum / (self._W - 1))
                          * (value - self._sum / (self._W - 1)) / self._W)
        self._sum += value

    cpdef compress_buckets(self):
        cdef int k, i, n_id
        cdef double n1, n2, u1, u2, incVariance, value, variance
        cdef AdWinListNode cursor, nextNode

        i = 0
        n_id = 0
        cursor = self._buckets.head
        while True:
            k = cursor._size
            if k == self._MAXBUCKETS+1:
                nextNode = self._buckets[n_id + 1]
                # nextNode = self.cursor._next
                if nextNode is None:
                    self._buckets.add_tail()
                    # nextNode = cursor._next
                    nextNode = self._buckets[n_id + 1]
                    self._last_bucket += 1
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor._sum[0]/n1
                u2 = cursor._sum[1]/n2
                incVariance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
                value = cursor._sum[0] + cursor._sum[1]
                variance = cursor._variance[1] + incVariance
                nextNode.add_back(value=value, var=variance)
                self._bucket_n -= 1
                cursor.drop_front(2)
                if nextNode._size <= self._MAXBUCKETS:
                    break
            else:
                break
            n_id += 1
            cursor = self._buckets[n_id]
            # cursor = cursor._next
            i += 1
            if cursor is None:
                break

    cpdef bint drop_check_drift(self):
        """ Detects concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from
        Time-Changing Data with Adaptive Windowing'. The general idea is to keep
        statistics from a window of variable size while detecting concept drift.

        This function is responsible for analysing different cutting points in
        the sliding window, to verify if there is a significant change in concept.

        Returns
        -------
        change : bool
            Whether change was detected or not

        Notes
        -----
        If change was detected, one should verify the new window size, by reading
        the width property.

        """
        cdef int k, min_length
        cdef bint change, quit
        cdef double n0, n1, u0, u1
        cdef AdWinListNode it

        change = False
        quit = False
        it = None

        self._mint_time += 1
        if (self._mint_time % self._mint_clock == 0 and
           self._W > self._min_long_wind):
            blnTalla = True

            while blnTalla:
                blnTalla = False
                quit = False
                n0 = 0.0
                n1 = self._W
                u0 = 0.0
                u1 = self._sum
                it = self._buckets.tail
                i = self._last_bucket

                while not quit and it:
                    for k in range(it._size - 1):
                        if i == 0 and k == it._size-1:
                            quit = True
                            break
                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += it._sum[k]
                        u1 -= it._sum[k]
                        min_lenght = 5
                        if (n0 >= min_lenght and n1 >= min_lenght
                           and self.cut_expression(n0, n1, u0, u1)):
                            blnTalla = True
                            change = True
                            if self._W > 0:
                                self.delete_element()
                                quit = True
                                break
                    it = it._prev
                    i -= 1

        return change

    cpdef double delete_element(self):
        """ Delete an Item from the bucket list.

        Deletes the last Item and updates relevant statistics kept by ADWIN.

        Returns
        -------
        int
            The bucket size from the updated bucket

        """
        cdef double n1, u1, inc_var
        cdef AdWinListNode tail

        tail = self._buckets.tail
        n1 = self.bucket_size(self._last_bucket)
        self._W -= n1
        self._sum -= tail._sum[0]
        u1 = tail._sum[0] / n1
        inc_var = ((tail._variance[0] + n1 * self._W *
                    (u1 - self._sum / self._W) * (u1 - self._sum / self._W))
                   / (n1 + self._W))
        self._var -= inc_var
        tail.drop_front()
        self._bucket_n -= 1
        if tail._size == 0:
            self._buckets.remove_tail()
            self._last_bucket -= 1

        return n1

    cpdef double cut_expression(self, double n0, double n1,
                                double u0, double u1):
        cdef int min_lenght
        cdef double n, diff, v, dd, m, eps

        n = self._W
        diff = u0 / n0 - u1 / n1

        v = self._var / self._W
        dd = log(2.0 * log(n) / self._delta)

        min_length = 5
        m = (1 / (n0 - min_length + 1)) + (1 / (n1 - min_length + 1))
        # retard fix patch for numeric approximation error
        # (need to debug latter)
        try:
            eps = sqrt(2.0 * m * v * dd) + 2.0 / 3 * dd * m
        except ValueError:
            if m < 10e-15:
                m = 0
            if v < 10e-15:
                v = 0
            if dd < 10e-15:
                dd = 0
            eps = sqrt(2.0 * m * v * dd) + 2.0 / 3 * dd * m

        return fabs(diff) > eps

    cpdef int bucket_size(self, double row):
        return int(pow(2, row))


class ADWIN(ADWINC, DriftDetector):
    """ Adaptive Windowing method for concept drift detection.

    Parameters
    ----------
    delta : float (default=0.002)
        The delta parameter for the ADWIN algorithm.

    Notes
    -----
    ADWIN (ADaptive WINdowing) is an adaptive sliding window algorithm
    for detecting change, and keeping updated statistics about a data stream.
    ADWIN allows algorithms not adapted for drifting data, to be resistant
    to this phenomenon.

    The general idea is to keep statistics from a window of variable size while
    detecting concept drift.

    The algorithm will decide the size of the window by cutting the statistics'
    window at different points and analysing the average of some statistic over
    these two windows. If the absolute value of the difference between the two
    averages surpasses a pre-defined threshold, change is detected at that point
    and all data before that time is discarded.

    References:
        1.  Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive
            windowing." In Proceedings of the 2007 SIAM international conference on data mining,
            pp. 443-448. Society for Industrial and Applied Mathematics, 2007.

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

    """
    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        super(ADWIN, self).__init__(delta)
        DriftDetector.__init__(self)
        # default values affected by init_bucket()
        DriftDetector.reset(self)
        # adwin properties
        self.bucket_num_max = 0
        self._n_detections = 0

    def reset(self):
        """Reset the change detector.
        """
        self.__init__(delta=self.delta)

    def set_clock(self, clock):
        self._mint_clock = clock

    @property
    def _bucket_used_bucket(self):
        return self.bucket_num_max

    @property
    def width(self):
        return self._W

    @property
    def n_detections(self):
        return self._n_detections

    @property
    def total(self):
        return self._sum

    @property
    def variance(self):
        return self._var / self._W

    @property
    def estimation(self):
        if self._W == 0:
            return 0
        return self._sum / self._W

    @estimation.setter
    def estimation(self, value):
        pass

    @property
    def width_t(self):
        return self._mdbl_width

    def __init_buckets(self):
        """ Initialize the bucket's List and statistics

        Set all statistics to 0 and create a new bucket List.

        """
        self._buckets = AdWinList(self._MAXBUCKETS)
        self.last_bucket_row = 0
        self._sum = 0
        self._var = 0
        self._W = 0
        self._bucket_n = 0

    def update(self, value):
        """Update the change detector with a single data point.

        Apart from adding the element value to the window, by inserting it in
        the correct bucket, it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.

        Parameters
        ----------
        value: Input value

        Notes
        -----
        The value parameter can be any numeric value relevant to the analysis
        of concept change. For the learners in this framework we are using
        either 0's or 1's, that are interpreted as follows:
        0: Means the learners prediction was wrong
        1: Means the learners prediction was correct

        This function should be used at every new sample analysed.

        Returns
        -------
        tuple
            A tuple (drift, warning) where its elements indicate if a drift or a warning is
            detected.

        """
        bln_change = super(ADWIN, self).update(value)
        if bln_change:
            self._n_detections += 1

        return bln_change, self._in_warning_zone
