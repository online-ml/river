import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class EDDM(BaseDriftDetector):
    """ Early Drift Detection Method
    
    This detection method was created to improve the detection 
    rate of gradual concept drift in DDM, while keeping a good 
    performance against abrupt concept drift. The implementation 
    and complete method explanation can be found in Early Drift 
    Detection Method by Baena-García, Campo-Ávila, Fidalgo, Bifet, 
    Gavaldà and Morales-Bueno.
    
    This method works by keeping track of the average distance 
    between two errors instead of only the error rate. For this, 
    it's necessary to keep track of the running average distance 
    and the running standard deviation, as well as the maximum 
    distance and the maximum standard deviation.
    
    The algorithm works similarly to the DDM algorithm, by keeping 
    track of statistics only. It works with the running average 
    distance (pi') and the running standard deviation (si'), as 
    well as p'max and s'max, which are the values of pi' and si' 
    when (pi' + 2 * si') reaches its maximum.
    
    Like DDM, there are two threshold values that define the 
    borderline between no change, warning zone, and drift detected.
    These are as follows:
    if (pi' + 2 * si')/(p'max + 2 * s'max) < alpha -> Warning zone
    if (pi' + 2 * si')/(p'max + 2 * s'max) < beta -> Change detected
    
    Alpha and beta are set to 0.95 and 0.9, respectively.
    
    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.eddm import EDDM
    >>> eddm = EDDM()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an 
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to EDDM and verifying if drift occurred
    >>> for i in range(2000):
    ...     eddm.add_element(data_stream[i])
    ...     if eddm.detected_warning_zone():
    ...         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    ...     if eddm.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    
    """
    FDDM_OUTCONTROL = 0.9
    FDDM_WARNING = 0.95
    FDDM_MIN_NUM_INSTANCES = 30

    def __init__(self):
        super().__init__()
        self.m_num_errors = None
        self.m_min_num_errors = 30
        self.m_n = None
        self.m_d = None
        self.m_lastd = None
        self.m_mean = None
        self.m_std_temp = None
        self.m_m2s_max = None
        self.m_last_level = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.m_n = 1
        self.m_num_errors = 0
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_std_temp = 0.0
        self.m_m2s_max = 0.0
        self.estimation = 0.0

    def add_element(self, prediction):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        
        Returns
        -------
        EDDM
            self
        
        Notes
        -----
        After calling this method, to verify if change was detected or if  
        the learner is in the warning zone, one should call the super method 
        detected_change, which returns True if concept drift was detected and
        False otherwise.
         
        """

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False

        self.m_n += 1

        if prediction == 1.0:
            self.in_warning_zone = False
            self.delay = 0
            self.m_num_errors += 1
            self.m_lastd = self.m_d
            self.m_d = self.m_n - 1
            distance = self.m_d - self.m_lastd
            old_mean = self.m_mean
            self.m_mean = self.m_mean + (1.0*distance - self.m_mean) / self.m_num_errors
            self.estimation = self.m_mean
            self.m_std_temp = self.m_std_temp + (distance - self.m_mean) * (distance - old_mean)
            std = np.sqrt(self.m_std_temp/self.m_num_errors)
            m2s = self.m_mean + 2 * std

            if m2s > self.m_m2s_max:
                if self.m_n > self.FDDM_MIN_NUM_INSTANCES:
                    self.m_m2s_max = m2s
            else:
                p = m2s / self.m_m2s_max
                if (self.m_n > self.FDDM_MIN_NUM_INSTANCES) and \
                        (self.m_num_errors > self.m_min_num_errors) and \
                        (p < self.FDDM_OUTCONTROL):
                    self.in_concept_change = True

                elif (self.m_n > self.FDDM_MIN_NUM_INSTANCES) and \
                        (self.m_num_errors > self.m_min_num_errors) and \
                        (p < self.FDDM_WARNING):
                    self.in_warning_zone = True

                else:
                    self.in_warning_zone = False

    def get_info(self):
        return 'EDDM: min_num_errors: ' + str(self.m_min_num_errors) + \
               ' - n_samples: ' + str(self.m_n) + \
               ' - mean: ' + str(self.m_mean) + \
               ' - std_dev: ' + str(round(np.sqrt(self.m_std_temp/self.m_num_errors), 3))
