__author__ = 'Guilherme Matsumoto'

import numpy as np
from skmultiflow.data.base_instance_stream import BaseInstanceStream
from sklearn.datasets import make_regression


class RegressionGenerator(BaseInstanceStream):
    """ RegressionGenerator
    
    This generator creates a stream of samples for a regression problem. It 
    uses the make_regression function from scikit-learn, which creates a 
    batch setting regression problem. These samples are then sequentially 
    fed by the next_instance function.
    
    Parameters
    ----------
    n_samples: int (Default: 40000)
        Total amount of samples to generate.
    
    n_features: int (Default: 100)
        Number of features to generate.
        
    n_informative: int (Default: 10)
        Number of relevant features, in other words, the number of features 
        that influence the class label.
    
    n_targets: int (Default: 1)
        Number of targeting tasks to generate.
    
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.regression_generator import RegressionGenerator
    >>> # Setting up the stream
    >>> stream = RegressionGenerator(n_samples=40000, n_features=10, n_targets=4, n_informative=6)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_instance()
    (array([[ 0.08847145,  0.54167032, -0.26639876,  0.16901383, -1.23675074,  0.2191299,  0.37675925,  
    1.08785428,  1.34738876,  1.75185625]]), array([[ 43.5869813 ,  20.17934387, -89.84100678,  99.71772926]]))
    >>> # Retrieving 10 samples
    >>> stream.next_instance(10)
    (array([[ 1.00067062,  0.21315385,  0.71619444, -1.07405979, -0.44566047,
         0.21042344, -0.12013223,  1.52176296, -0.54644817, -0.22761552],
       [-0.11595306, -0.09317076,  1.6024751 ,  0.40429975, -1.45060292,
         1.21409322,  0.28001195,  1.24693445, -1.34851791, -1.49512122],
       [ 0.88309146,  1.90792011, -0.18723735,  0.6161133 , -1.49582992,
         1.81954813, -0.90094923,  1.03586027,  0.11026183,  1.12514587],
       [ 0.75196922,  0.64277909, -0.89305047,  0.52819719, -0.38798905,
         0.35649152, -1.57671827,  1.16010305,  0.58344281,  1.0759902 ],
       [ 0.88576633,  0.7858232 , -0.26126212, -1.10802188, -0.89619512,
        -0.44915036, -0.11700449,  0.54494865, -0.91151171,  1.50259135],
       [ 0.20054457, -0.84940252,  0.40505224,  0.23804963, -0.37821232,
         0.38167094, -0.45447389, -0.20675609, -0.22947645, -0.5346733 ],
       [-0.78212843,  0.15420883, -0.27916141,  1.56364308, -0.11299342,
         1.03975364,  0.22823241,  0.00348303, -0.25840339,  1.31217362],
       [-0.16351539,  0.60941347, -0.20294223,  0.66026152,  0.05045567,
         0.22316186,  0.93197562,  1.32542373, -0.04543523,  2.44725885],
       [-0.50325537, -1.22869527,  1.43718402, -2.05933559,  0.46698075,
         1.66436076,  0.31205451, -1.02400179, -0.23611307, -0.43686569],
       [ 1.10504135,  0.45413326,  1.09731459,  2.01807386,  1.61785921,
        -0.3835003 ,  0.88883791, -1.09485607,  0.07325024,  0.31294406]]), array([[ -47.6385461 ,  -43.88370836,   58.56857132,  -57.56014259],
       [  62.66644355, -119.4593376 ,   -7.46088743,  -78.88993459],
       [  42.96563939, -143.34991562,  -71.16390347,  -56.31677899],
       [ -17.30988744, -108.13263269,  -49.21158065,  -63.32182558],
       [ -87.45441879, -158.18131269,  -62.08270835, -116.75603504],
       [  11.89772626,  -50.86442522,   10.1538106 ,  -41.19856008],
       [ 112.33174356,  -11.03547029,  -54.04449607,   23.16505909],
       [ 106.53060383,   76.43868354,    0.4905551 ,   85.87867079],
       [-177.64080457,   33.09515266,   73.51070668,  -45.71507071],
       [ 352.84278173,  362.40102842,  332.56626222,  241.00957637]]))
    >>> stream.estimated_remaining_instances()
    39989
    >>> stream.has_more_instances()
    True
    
    """

    def __init__(self, n_samples=40000, n_features=100, n_informative=10, n_targets=1):
        super().__init__()
        self.X = None
        self.y = None
        self.num_samples = 0
        self.num_features = 0
        self.num_target_tasks = 0
        self.num_informative = 0
        self.instance_index = 0
        self.current_instance_y = None
        self.current_instance_x = None
        self.__configure(n_samples, n_features, n_informative, n_targets)

    def __configure(self, n_samples, n_features, n_informative, n_targets):
        """ __configure
        
        Uses the make_regression function from scikit-learn to generate a 
        regression problem. This problem will be kept in memory and provided 
        as demanded.
        
        Parameters
        ----------
        n_samples: int
            Total amount of samples to generate.
        
        n_features: int
            Number of features to generate.
            
        n_informative: int
            Number of relevant features, in other words, the number of features 
            that influence the class label.
        
        n_targets: int
            Number of targeting tasks to generate.
        
        """
        self.X, self.y = make_regression(n_samples=n_samples, n_features=n_features,
                                         n_informative=n_informative, n_targets=n_targets)
        self.y.resize((self.y.size, n_targets))
        self.num_samples = n_samples
        self.num_features = n_features
        self.num_target_tasks = n_targets
        self.num_informative = n_informative

    def estimated_remaining_instances(self):
        return (self.num_samples - self.instance_index)

    def has_more_instances(self):
        return (self.num_samples - self.instance_index > 0)

    def next_instance(self, batch_size=1):
        """ next_instance
        
        Returns batch_size samples from the generated regression problem.
        
        Parameters
        ----------
        batch_size: int
            The number of sample to return.
            
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested.
        
        """
        self.instance_index += batch_size
        try:

            self.current_instance_x = self.X[self.instance_index - batch_size:self.instance_index, :]
            self.current_instance_y = self.y[self.instance_index - batch_size:self.instance_index, :]
            if self.num_target_tasks < 2:
                self.current_instance_y = self.current_instance_y.flatten()

        except IndexError:
            self.current_instance_x = None
            self.current_instance_y = None
        return (self.current_instance_x, self.current_instance_y)

    def is_restartable(self):
        return True

    def restart(self):
        pass

    def has_more_mini_batch(self):
        pass

    def get_num_nominal_attributes(self):
        pass

    def get_num_numerical_attributes(self):
        pass

    def get_num_values_per_nominal_attribute(self):
        pass

    def get_num_attributes(self):
        return self.num_features

    def get_num_targets(self):
        return self.num_target_tasks

    def get_attributes_header(self):
        pass

    def get_classes_header(self):
        pass

    def get_last_instance(self):
        return self.current_instance_x, self.current_instance_y

    def prepare_for_use(self):
        pass

    def get_plot_name(self):
        return 'Regression dataset'

    def get_classes(self):
        return np.unique(self.y)

    def get_class_type(self):
        return 'stream'

    def get_info(self):
        return 'RegressionGenerator: n_samples: ' + str(self.num_samples) + \
               ' - n_features: ' + str(self.num_features) + \
               ' - n_informative: ' + str(self.num_informative) + \
               ' - n_targets: ' + str(self.num_target_tasks)

    def get_num_targeting_tasks(self):
        return self.num_target_tasks
