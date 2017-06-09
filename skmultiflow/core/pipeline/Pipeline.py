__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.utils.Utils import dict_to_list


class Pipeline:
    '''
        class Pipeline
        this is not gonna work if the functions that are to be execute are not well defined.
        There's at least one classifier in the pipeline. There can be a maximum of one evaluator, and as many
        transforms as one should need.
    '''
    def __init__(self, steps, containsEvaluator = False):
        '''
            __init__(self, steps, containsEvaluator)
            --------------------------------
            
            Creates a pipeline, which will execute the functions received in the dict.
            :param step: dictionary of functions to execute in order.
        '''

        #default values
        self.num_transforms = 0

        self.steps = dict_to_list(steps)
        self.keys = steps.keys()
        self.has_evaluator = containsEvaluator
        self.configure()
        pass

    def configure(self):
        self.num_transforms = (len(self.steps) - 1) if not self.has_evaluator else (len(self.steps) - 2)
        pass

    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

    def partial_fit(self, X, y):
        pass

    def fit_predict(self, X, y):
        pass

    def fit_transform(self, X, y = None):
        pass

    def execute_cycle(self, *args):
        '''
        
        :param args: matrix of the arguments to all step functions 
        :return: 
        '''
        pass

    def run(self, stream):
        pass