from . import base

class Histogram(base.RunningStatistic):

    def __init__(self,bins = 80):
        self.bins = bins
    
    @property
    def name(self):
        return 'histogram'
    
    def update(self, x):
        return self

    def get(self):
    
        return self.histogram 