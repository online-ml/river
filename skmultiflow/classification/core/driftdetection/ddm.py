__author__ = 'Guilherme Matsumoto'

import numpy as np

from skmultiflow.classification.core.driftdetection.base_drift_detector import BaseDriftDetector


class DDM(BaseDriftDetector):
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
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1
        self.miss_std = 0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def add_element(self, prediction):
        '''
        
        :param prediction: 0 if there was no misclassification, 1 if there was a misclassification happened
        :return: 
        '''
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / (1. * self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / self.sample_count)
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if (self.sample_count < self.min_instances):
            return None

        if (self.miss_prob + self.miss_std <= self.miss_prob_sd_min):
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if (self.sample_count > self.min_instances) and (self.miss_prob + self.miss_std > self.miss_prob_min + 3*self.miss_sd_min):
            self.in_concept_change = True
        elif self.miss_prob + self.miss_std > self.miss_prob_min + 2*self.miss_sd_min:
            self.in_warning_zone = True
        else:
            self.in_warning_zone = False

        return None

    def get_info(self):
        return 'Not implemented'
