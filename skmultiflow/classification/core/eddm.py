__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.classification.core.base_drift_detector import BaseDriftDetector


class EDDM(BaseDriftDetector):

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
                if (self.m_n > self.FDDM_MIN_NUM_INSTANCES) and (self.m_num_errors > self.m_min_num_errors) and (
                    p < self.FDDM_OUTCONTROL):
                    self.in_concept_change = True

                elif (self.m_n > self.FDDM_MIN_NUM_INSTANCES) and (self.m_num_errors > self.m_min_num_errors) and (
                    p < self.FDDM_WARNING):
                    self.in_warning_zone = True

                else:
                    self.in_warning_zone = False


    def get_info(self):
        return 'Not implemented.'