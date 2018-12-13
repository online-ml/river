import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class DDM(BaseDriftDetector):
    """ Drift Detection Method
    
    This concept change detection method is based on the PAC learning model 
    premise, that the learner's error rate will decrease as the number of 
    analysed samples increase, as long as the data distribution is 
    stationary.
    
    If the algorithm detects an increase in the error rate, that surpasses 
    a calculated threshold, either change is detected or the algorithm will 
    warn the user that change may occur in the near future, which is called 
    the warning zone.
    
    The detection threshold is calculated in function of two statistics, 
    obtained when (pi + si) is minimum: 
    pmin: The minimum recorded error rate.
    smin: The minimum recorded standard deviation.
    
    At instant i, the detection algorithm uses:
    pi: The error rate at instant i.
    si: The standard deviation at instant i.
    
    The conditions for entering the warning zone and detecting change are 
    as follows:
    if pi + si >= pmin + 2 * smin -> Warning zone
    if pi + si >= pmin + 3 * smin -> Change detected
    
    Parameters
    ----------
    min_num_instances: int
        The minimum required number of analyzed samples so change can be 
        detected. This is used to avoid false detections during the early 
        moments of the detector, when the weight of one sample is important.
        
    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection import DDM
    >>> ddm = DDM()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an 
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to DDM and verifying if drift occurred
    >>> for i in range(2000):
    ...     ddm.add_element(data_stream[i])
    ...     if ddm.detected_warning_zone():
    ...         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    ...     if ddm.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    
    """

    def __init__(self, min_num_instances=30):
        super().__init__()
        self.min_instances = min_num_instances
        self.sample_count = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1
        self.miss_std = 0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def add_element(self, prediction):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        
        Notes
        -----
        After calling this method, to verify if change was detected or if  
        the learner is in the warning zone, one should call the super method 
        detected_change, which returns True if concept drift was detected and
        False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / (1. * self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / self.sample_count)
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            pass

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if (self.sample_count > self.min_instances) and (self.miss_prob + self.miss_std >
                                                         self.miss_prob_min + 3*self.miss_sd_min):
            self.in_concept_change = True

        elif self.miss_prob + self.miss_std > self.miss_prob_min + 2 * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False

    def get_info(self):
        return 'DDM: min_num_instances: ' + str(self.min_instances) + \
               ' - sample_count: ' + str(self.sample_count) + \
               ' - error_rate: ' + str(self.miss_prob) + \
               ' - std_dev: ' + str(self.miss_std) + \
               ' - error_rate_min: ' + str(self.miss_prob_min) + \
               ' - std_dev_min: ' + str(self.miss_sd_min)
