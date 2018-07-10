from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class PageHinkley(BaseDriftDetector):
    """ Page Hinkley change detector
    
    This change detection method works by computing the observed 
    values and their mean up to the current moment. Page Hinkley 
    won't output warning zone warnings, only change detections. 
    The method works by means of the Page Hinkley test. In general 
    lines it will detect a concept drift if the observed mean at 
    some instant is greater then a threshold value lambda.
    
    Parameters
    ----------
    min_num_instances: int
        The minimum number of instances before detecting change.
    delta: float
        The delta factor for the Page Hinkley test.
    _lambda: int
        The change detection threshold.
    alpha: float
        The forgetting factor, used to weight the observed value 
        and the mean.
    
    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection import PageHinkley
    >>> ph = PageHinkley()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 2000
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>> # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    >>> for i in range(2000):
    ...     ph.add_element(data_stream[i])
    ...     if ph.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    
    """
    def __init__(self, min_num_instances=30, delta=0.005, _lambda=50, alpha=1-0.0001):
        super().__init__()
        self.min_instances = min_num_instances
        self.delta = delta
        self._lambda = _lambda
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def add_element(self, x):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.
        
        Notes
        -----
        After calling this method, to verify if change was detected, one 
        should call the super method detected_change, which returns True 
        if concept drift was detected and False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / 1.0 * self.sample_count
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self._lambda:
            self.in_concept_change = True

    def get_info(self):
        return 'PageHinkley: min_num_instances: ' + str(self.min_instances) + \
               ' - delta: ' + str(self.delta) + \
               ' - lambda: ' + str(self._lambda) + \
               ' - alpha: ' + str(self.alpha)
