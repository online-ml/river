import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ADWIN(BaseDriftDetector):
    """ Adaptive Windowing method for concept drift detection.
    
    Parameters
    ----------
    delta : float (default=0.002)
        The delta parameter for the ADWIN algorithm.

    Notes
    -----
    ADWIN [1]_ (ADaptive WINdowing) is an adaptive sliding window algorithm
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

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavalda. "Learning from time-changing data with adaptive windowing."
       In Proceedings of the 2007 SIAM international conference on data mining, pp. 443-448.
       Society for Industrial and Applied Mathematics, 2007.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.adwin import ADWIN
    >>> adwin = ADWIN()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 2000
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>> # Adding stream elements to ADWIN and verifying if drift occurred
    >>> for i in range(2000):
    ...     adwin.add_element(data_stream[i])
    ...     if adwin.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    
    """
    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
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

        self.mdbl_delta = .002
        self.mint_time = 0
        self.mdbl_width = 0

        self.detect = 0
        self._n_detections = 0
        self.detect_twice = 0
        self.mint_clock = 32

        self.bln_bucket_deleted = False
        self.bucket_num_max = 0
        self.mint_min_window_length = 5
        super().reset()

    def reset(self):
        """ Reset detectors
        
        Resets statistics and adwin's window.
        
        Returns
        -------
        ADWIN 
            self
        
        """
        self.__init__(delta=self.delta)

    def get_change(self):
        """ Get drift
        
        Returns
        -------
        bool
            Whether or not a drift occurred 
        
        """
        return self.bln_bucket_deleted

    def reset_change(self):
        self.bln_bucket_deleted = False

    def set_clock(self, clock):
        self.mint_clock = clock

    def detected_warning_zone(self):
        return False

    @property
    def _bucket_used_bucket(self):
        return self.bucket_num_max

    @property
    def width(self):
        return self._width

    @property
    def n_detections(self):
        return self._n_detections

    @property
    def total(self):
        return self._total

    @property
    def variance(self):
        return self._variance / self._width

    @property
    def estimation(self):
        if self._width == 0:
            return 0
        return self._total / self._width

    @estimation.setter
    def estimation(self, value):
        pass

    @property
    def width_t(self):
        return self.mdbl_width

    def __init_buckets(self):
        """ Initialize the bucket's List and statistics
        
        Set all statistics to 0 and create a new bucket List.
        
        """
        self.list_row_bucket = List()
        self.last_bucket_row = 0
        self._total = 0
        self._variance = 0
        self._width = 0
        self.bucket_number = 0

    def add_element(self, value):
        """ Add a new element to the sample window.
        
        Apart from adding the element value to the window, by inserting it in 
        the correct bucket, it will also update the relevant statistics, in 
        this case the total sum of all values, the window width and the total 
        variance.
        
        Parameters
        ----------
        value: int or float (a numeric value)
         
        Notes
        -----
        The value parameter can be any numeric value relevant to the analysis 
        of concept change. For the learners in this framework we are using 
        either 0's or 1's, that are interpreted as follows:
        0: Means the learners prediction was wrong
        1: Means the learners prediction was correct
        
        This function should be used at every new sample analysed.
         
        """
        self._width += 1
        self.__insert_element_bucket(0, value, self.list_row_bucket.first)
        incremental_variance = 0

        if self._width > 1:
            incremental_variance = (self._width - 1) * (value - self._total / (self._width - 1)) * \
                                   (value - self._total / (self._width - 1)) / self._width

        self._variance += incremental_variance
        self._total += value
        self.__compress_buckets()

    def __insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

    @staticmethod
    def bucket_size(row):
        return np.power(2, row)

    def delete_element(self):
        """ Delete an Item from the bucket list.
         
        Deletes the last Item and updates relevant statistics kept by ADWIN.
        
        Returns
        -------
        int
            The bucket size from the updated bucket
        
        """
        node = self.list_row_bucket.last
        n1 = self.bucket_size(self.last_bucket_row)
        self._width -= n1
        self._total -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = node.get_variance(0) + n1 * self._width * (u1 - self._total / self._width) * \
                               (u1 - self._total / self._width) / (n1 + self._width)
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
                n1 = self.bucket_size(i)
                n2 = self.bucket_size(i)
                u1 = cursor.get_total(0)/n1
                u2 = cursor.get_total(1)/n2
                incremental_variance = n1 * n2 * (u1 - u2) / (n1 + n2)
                next_node.insert_bucket(cursor.get_total(0) + cursor.get_total(1), cursor.get_variance(1)
                                        + incremental_variance)
                self.bucket_number += 1
                cursor.compress_bucket_row(2)

                if next_node.bucket_size_row <= self.MAX_BUCKETS:
                    break
            else:
                break

            cursor = cursor.get_next_item()
            i += 1

    def detected_change(self):
        """ Detects concept change in a drifting data stream.
        
        The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from 
        Time-Changing Data with Adaptive Windowing'. The general idea is to keep 
        statistics from a window of variable size while detecting concept drift.
         
        This function is responsible for analysing different cutting points in 
        the sliding window, to verify if there is a significant change in concept.
        
        Returns
        -------
        bln_change : bool
            Whether change was detected or not
            
        Notes
        -----
        If change was detected, one should verify the new window size, by reading 
        the width property.
        
        """
        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (self.width > self.mint_min_window_longitude):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self._width
                u0 = 0
                u1 = self.total
                v0 = 0
                v1 = self._variance
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket.last
                i = self.last_bucket_row

                while (not bln_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row - 1):
                        n2 = self.bucket_size(i)
                        u2 = cursor.get_total(k)

                        if n0 > 0:
                            v0 += cursor.get_variance(k) + 1. * n0 * n2 * (u0/n0 - u2/n2) * (u0/n0 - u2/n2) / (n0 + n2)

                        if n1 > 0:
                            v1 -= cursor.get_variance(k) + 1. * n1 * n2 * (u1/n1 - u2/n2) * (u1/n1 - u2/n2) / (n1 + n2)

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += cursor.get_total(k)
                        u1 -= cursor.get_total(k)

                        if (i == 0) and (k == cursor.bucket_size_row - 1):
                            bln_exit = True
                            break

                        abs_value = 1. * ((u0/n0) - (u1/n1))
                        if (n1 >= self.mint_min_window_length) and (n0 >= self.mint_min_window_length)\
                                and (self.__bln_cut_expression(n0, n1, u0, u1, v0, v1, abs_value, self.delta)):
                            bln_bucket_deleted = True
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            if self.width > 0:
                                n0 -= self.delete_element()
                                bln_exit = True
                                break

                    cursor = cursor.get_previous()
                    i -= 1
        self.mdbl_width += self.width
        if bln_change:
            self._n_detections += 1
        self.in_concept_change = bln_change
        return bln_change

    def __bln_cut_expression(self, n0, n1, u0, u1, v0, v1, abs_value, delta):
        n = self.width
        dd = np.log(2*np.log(n)/delta)
        v = self.variance
        m = (1. / (n0 - self.mint_min_window_length + 1)) + (1. / (n1 - self.mint_min_window_length + 1))
        epsilon = np.sqrt(2 * m * v * dd) + 1. * 2 / 3 * dd * m
        return np.absolute(abs_value) > epsilon


class List(object):
    """ A linked list object for ADWIN algorithm.
    
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
    """ Item to be used by the List object.
    
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
        self.bucket_total = np.zeros(self.max_buckets+1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets+1, dtype=float)
        self.reset()

    def reset(self):
        """ Reset the algorithm's statistics and window
        
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
            self.bucket_total[i-num_deleted] = self.bucket_total[i]
            self.bucket_variance[i-num_deleted] = self.bucket_variance[i]

        for i in range(1, num_deleted+1):
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
