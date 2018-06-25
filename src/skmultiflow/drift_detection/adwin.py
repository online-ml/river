import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.core.base_object import BaseObject


class ADWIN(BaseDriftDetector):
    """ ADWIN change detector for concept change detection
    
    ADWIN stands for Adaptive Windowing, and is an adaptive sliding window 
    algorithm for detecting change, and keeping updated statistics about 
    a data stream. Allows algorithms not adapted for drifting data, to be 
    more resistant to this phenomenon.
    
    The ADWIN algorithm is described in Bifet and Gavaldà's 'Learning from 
    Time-Changing Data with Adaptive Windowing'. The general idea is to keep 
    statistics from a window of variable size while detecting concept drift. 
    
    The algorithm will decide the size of the window by cutting the statistics' 
    window at different points and analysing the average of some statistic over 
    these two windows. If the absolute value of the difference between the two 
    averages surpasses a pre-defined threshold, change is detected at that point 
    and all data before that time is discarded. 
    
    Parameters
    ----------
    delta : float
        The delta parameter for the ADWIN algorithm.
    clock : int
        The base clock value for the ADWIN algorithm.
        
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
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    
    """
    MAXBUCKETS = 5

    def __init__(self, delta=.002, clock = None):
        super().__init__()
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_row = 0
        self.list_row_bucket = None
        self.TOTAL = 0
        self.VARIANCE = 0
        self.WIDTH = 0
        self.bucket_number = 0

        self.__init_buckets()

        # other default values
        self.mint_min_window_longitude = 10

        self.mdbl_delta = .002
        self.mint_time = 0
        self.mdbl_width = 0

        self.detect = 0
        self.num_detections = 0
        self.detect_twice = 0
        self.mint_clock = 32 if clock is None else clock

        self.bln_bucket_deleted = False
        self.bucket_num_max = 0
        self.mint_min_window_length = 5
        self.reset()

    def reset(self):
        """ Reset detectors
        
        Resets statistics and adwin's window.
        
        Returns
        -------
        ADWIN 
            self
        
        """
        super().reset()

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
    def _width(self):
        return self.WIDTH

    @property
    def _number_detections(self):
        return self.num_detections

    @property
    def _total(self):
        return self.TOTAL

    @property
    def _variance(self):
        return self.VARIANCE / self.WIDTH

    @property
    def _estimation(self):
        if self.WIDTH == 0:
            return 0
        return self.TOTAL / self.WIDTH

    @property
    def _width_t(self):
        return self.mdbl_width

    def __init_buckets(self):
        """ Initialize the bucket's List and statistics
        
        Set all statistics to 0 and create a new bucket List.
        
        """
        self.list_row_bucket = List()
        self.last_bucket_row = 0
        self.TOTAL = 0
        self.VARIANCE = 0
        self.WIDTH = 0
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
            For most of scikit-multiflow learners these values are either 
            1 or 0.
         
        Notes
        -----
        The value parameter can be any numeric value relevant to the analysis 
        of concept change. For the learners in this framework we are using 
        either 0's or 1's, that are interpreted as follows:
        0: Means the learners prediction was wrong
        1: Means the learners prediction was correct
        
        This function should be used at every new sample analysed.
         
        """
        self.WIDTH += 1
        self.__insert_element_bucket(0, value, self.list_row_bucket._first)
        incremental_variance = 0

        if self.WIDTH > 1:
            incremental_variance = (self.WIDTH - 1) * (value - self.TOTAL / (self.WIDTH - 1)) * \
                                   (value - self.TOTAL / (self.WIDTH - 1)) / self.WIDTH

        self.VARIANCE += incremental_variance
        self.TOTAL += value
        self.__compress_buckets()

    def __insert_element_bucket(self, variance, value, node):
        node.insert_bucket(value, variance)
        self.bucket_number += 1

        if self.bucket_number > self.bucket_num_max:
            self.bucket_num_max = self.bucket_number

    def bucket_size(self, row):
        return np.power(2, row)

    def delete_element(self):
        """ Delete an Item from the bucket list.
         
        Deletes the last Item and updates relevant statistics kept by ADWIN.
        
        Returns
        -------
        int
            The bucket size from the updated bucket
        
        """
        node = self.list_row_bucket._last
        n1 = self.bucket_size(self.last_bucket_row)
        self.WIDTH -= n1
        self.TOTAL -= node.get_total(0)
        u1 = node.get_total(0) / n1
        incremental_variance = node.get_variance(0) + n1 * self.WIDTH * (u1 - self.TOTAL / self.WIDTH) * \
                                                      (u1 - self.TOTAL / self.WIDTH) / (n1 + self.WIDTH)
        self.VARIANCE -= incremental_variance
        node.remove_bucket()
        self.bucket_number -= 1

        if node.bucket_size_row == 0:
            self.list_row_bucket.remove_from_tail()
            self.last_bucket_row -= 1

        return n1

    def __compress_buckets(self):
        next_node = None
        cursor = self.list_row_bucket._first
        i = 0
        while (cursor is not None):
            k = cursor.bucket_size_row
            if k == self.MAXBUCKETS + 1:
                next_node = cursor._next
                if next_node is None:
                    self.list_row_bucket.add_to_tail()
                    next_node = cursor._next
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

                if next_node.bucket_size_row <= self.MAXBUCKETS:
                    break
            else:
                break

            cursor = cursor._next
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
        the _width property.
        
        """
        bln_change = False
        bln_exit = False
        bln_bucket_deleted = False
        self.mint_time += 1
        n0 = 0
        if (self.mint_time % self.mint_clock == 0) and (self._width > self.mint_min_window_longitude):
            bln_reduce_width = True
            while bln_reduce_width:
                bln_reduce_width = not bln_reduce_width
                bln_exit = False
                n0 = 0
                n1 = self.WIDTH
                u0 = 0
                u1 = self._total
                v0 = 0
                v1 = self.VARIANCE
                n2 = 0
                u2 = 0
                cursor = self.list_row_bucket._last
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
                        if (n1 >= self.mint_min_window_length) and (n0 >= self.mint_min_window_length) and (
                        self.__bln_cut_expression(n0, n1, u0, u1, v0, v1, abs_value, self.delta)):
                            bln_bucket_deleted = True
                            self.detect = self.mint_time
                            if self.detect == 0:
                                self.detect = self.mint_time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.mint_time

                            bln_reduce_width = True
                            bln_change = True
                            if self._width > 0:
                                n0 -= self.delete_element()
                                bln_exit = True
                                break

                    cursor = cursor._previous
                    i -= 1
        self.mdbl_width += self._width
        if bln_change:
            self.num_detections += 1
        self.in_concept_change = bln_change
        return bln_change

    def __bln_cut_expression(self, n0, n1, u0, u1, v0, v1, abs_value, delta):
        n = self._width
        dd = np.log(2*np.log(n)/delta)
        v = self._variance
        m = (1. / (n0 - self.mint_min_window_length + 1)) + (1. / (n1 - self.mint_min_window_length + 1))
        epsilon = np.sqrt(2 * m * v *dd) + 1. * 2 / 3 * dd * m
        return (np.absolute(abs_value) > epsilon)

    def get_info(self):
        return 'ADWIN: delta: ' + str(self.delta) + \
               ' - clock: ' + str(self.mint_clock) + \
               ' - total: ' + str(self.TOTAL) + \
               ' - variance: ' + str(self.VARIANCE) + \
               ' - width: ' + str(self.WIDTH) + \
               ' - time: ' + str(self.mint_time) + \
               ' - num_detections: ' + str(self.num_detections)


class List(BaseObject):
    """ A linked list object for ADWIN algorithm.
    
    Used for storing ADWIN's bucket list. Is composed of Item objects. 
    Acts as a linked list, where each element points to its predecessor 
    and successor. 
    
    Attributes
    ----------
    _size: int 
        Read-only attribute to access the List's count
    _first: Item object
        The first bucket in the list
    _last: Item object
        The last bucket in the list
        
    """
    def __init__(self):
        super().__init__()
        self.count = None
        self.first = None
        self.last = None
        self.reset()
        self.add_to_head()

    def reset(self):
        self.count = 0
        self.first = None
        self.last = None

    def add_to_head(self):
        self.first = Item(self.first, None)
        if self.last is None:
            self.last = self.first

    def remove_from_head(self):
        self.first = self.first._next
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
        self.last = self.last._previous
        if self.last is not None:
            self.last.set_next(None)
        else:
            self.first = None
        self.count -=1

    @property
    def _first(self):
        return self.first

    @property
    def _last(self):
        return self.last

    @property
    def _size(self):
        return self.count

    def get_info(self):
        return 'List: count: ' + str(self.count)

    def get_class_type(self):
        return 'data_structure'


class Item(BaseObject):
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
            previous_item.next = self
        self.bucket_size_row = None
        self.max_buckets = ADWIN.MAXBUCKETS
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
        for i in range(ADWIN.MAXBUCKETS+1):
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
        for i in range(num_deleted, ADWIN.MAXBUCKETS+1):
            self.bucket_total[i-num_deleted] = self.bucket_total[i]
            self.bucket_variance[i-num_deleted] = self.bucket_variance[i]

        for i in range(1, num_deleted+1):
            self.__clear_buckets(ADWIN.MAXBUCKETS - i + 1)

        self.bucket_size_row -= num_deleted

    @property
    def _next(self):
        return self.next

    def set_next(self, next):
        self.next = next

    @property
    def _previous(self):
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

    def get_info(self):
        return 'Item: bucket_size_row: ' + str(self.bucket_size_row) + \
               ' - max_buckets: ' + str(self.max_buckets) + \
               ' - bucket_total: ' + str(self.bucket_total) + \
               ' - bucket_variance: ' + str(self.bucket_variance)

    def get_class_type(self):
        return 'data_structure'