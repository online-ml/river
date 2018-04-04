__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.utils import pseudo_random_processes as prp
from skmultiflow.core.base_object import BaseObject
import numpy as np


class RandomRBFGenerator(BaseInstanceStream, BaseObject):
    """ RandomRBFGenerator
    
    This generator produces a radial basis function stream.
    
    A number of centroids, having a random central position, a standard 
    deviation, a class label and weight, are generated. A new sample is 
    created by choosing one of the centroids at random, taking into 
    account their weights, and offsetting the attributes at a random 
    direction from the centroid's center. The offset length is drawn 
    from a Gaussian distribution.
      
    This process will create a normally distributed hypersphere of samples 
    on the surrounds of each centroid.
    
    Parameters
    ---------
    model_seed: int (Default: 21)
        The seed to be used by the model random generator.
        
    instance_seed: int (Default: 5)
        The seed to be used by the instance random generator.
        
    num_classes: int (Default: 2)
        The number of class labels to generate.
        
    num_att: int (Default: 10)
        The total number of attributes to generate.
        
    num_centroids: int (Default: 50)
        The total number of centroids to generate.
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.random_rbf_generator import RandomRBFGenerator
    >>> # Setting up the stream
    >>> stream = RandomRBFGenerator(model_seed=99, instance_seed=50, num_classes=4, num_att=10, num_centroids=50)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_instance()
    (array([[ 0.44952282,  1.09201096,  0.34778443,  0.92181679,  0.19503463,
         0.28834419,  0.8293168 ,  0.26847952,  0.8096243 ,  0.23850379]]), array([[ 3.]]))
    >>> # Retrieving 10 samples
    >>> stream.next_instance(10)
    (array([[ 0.70374896,  0.65752835,  0.20343463,  0.56136917,  0.76659286,
         0.61081231,  0.60453064,  0.88734577, -0.04244631,  0.09146432],
       [ 0.27797196,  0.05640135,  0.80946171,  0.60572837,  0.95080656,
         0.25512099,  0.73992469,  0.33917142,  0.17104577,  0.79283295],
       [ 0.33696167,  0.10923638,  0.85987231,  0.61868598,  0.85755211,
         0.19469184,  0.66750447,  0.27684404,  0.1554274 ,  0.76262286],
       [ 0.71886223,  0.23078927,  0.45013806,  0.03019141,  0.42679505,
         0.03841721,  0.34318517,  0.11769923,  0.9644654 ,  0.01635577],
       [-0.01849262,  0.92570731,  0.87564868,  0.49372553,  0.39717634,
         0.46697609,  0.41329831,  0.27652149,  0.12724455,  0.24658299],
       [ 0.81850217,  0.87228851,  0.18873385, -0.04254749,  0.06942877,
         0.55567756,  0.97660009,  0.0273206 ,  0.67995834,  0.49135807],
       [ 0.69888163,  0.61994977,  0.43074298,  0.27526838,  0.69566798,
         0.91059369,  0.04680901,  0.50453698,  0.61394089,  0.92275292],
       [ 1.01929588,  0.80181051,  0.50547533,  0.14715636,  0.42889167,
         0.61513174,  0.21752655, -0.52958207,  1.35091672,  0.38769673],
       [ 0.37738633,  0.60922205,  0.64216064,  0.90009707,  0.91787083,
         0.36189554,  0.35438165,  0.28510134,  0.55301333,  0.21450072],
       [ 0.62185359,  0.75178244,  1.00436662,  0.24412816,  0.41070861,
         0.52547739,  0.50978735,  0.79445216,  0.77589569,  0.16214271]]), array([[ 3.],
       [ 3.],
       [ 3.],
       [ 2.],
       [ 3.],
       [ 2.],
       [ 0.],
       [ 2.],
       [ 0.],
       [ 2.]]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.estimated_remaining_instances()
    -1
    >>> stream.has_more_instances()
    True
    
    """

    def __init__(self, model_seed=21, instance_seed=5, num_classes=2, num_att=10, num_centroids=50):
        super().__init__()

        # Default values
        self.num_numerical_attributes = 10
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.current_instance_x = None
        self.current_instance_y = None
        self.model_seed = 21
        self.instance_seed = 5
        self.num_classes = 2
        self.num_centroids = 50
        self.centroids = None
        self.centroid_weights = None
        self.instance_random = None
        self.attributes_header = None
        self.class_header = None

        self.__configure(model_seed, instance_seed, num_classes, num_att, num_centroids)

    def __configure(self, model_seed, instance_seed, num_classes, num_att, num_centroids):
        self.model_seed = model_seed
        self.instance_seed = instance_seed
        self.num_classes = num_classes
        self.num_numerical_attributes = num_att
        self.num_centroids = num_centroids
        self.instance_random = np.random
        self.instance_random.seed(self.instance_seed)

        self.class_header = ["class"]
        self.attributes_header = []
        for i in range(self.num_numerical_attributes):
            self.attributes_header.append("att_num_" + str(i))

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size=1):
        """ next_instance
        
        Return batch_size samples generated by choosing a centroid at 
        random and randomly offsetting its attributes so that it is 
        placed inside the hypersphere of that centroid.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested. 
        
        """
        data = np.zeros([batch_size, self.num_numerical_attributes + 1])
        num_atts = self.num_numerical_attributes
        for j in range(batch_size):
            centroid_aux = self.centroids[prp.random_index_based_on_weights(self.centroid_weights,
                                                                            self.instance_random)]
            att_vals = []
            magnitude = 0.0
            for i in range(num_atts):
                att_vals.append((self.instance_random.rand() * 2.0) - 1.0)
                magnitude += att_vals[i]*att_vals[i]
            magnitude = np.sqrt(magnitude)
            desired_mag = self.instance_random.normal() * centroid_aux.std_dev
            scale = desired_mag/magnitude
            for i in range(num_atts):
                data[j, i] = centroid_aux.centre[i] + att_vals[i]*scale
            data[j, num_atts] = centroid_aux.class_label
        self.current_instance_x = data[:, :num_atts]
        self.current_instance_y = data[:, num_atts:]
        return self.current_instance_x, self.current_instance_y

    def prepare_for_use(self):
        self.restart()

    def is_restartable(self):
        return True

    def restart(self):
        self.generate_centroids()
        self.instance_random.seed(self.instance_seed)

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)

    def get_num_targets(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.class_header

    def get_last_instance(self):
        return self.current_instance_x, self.current_instance_y

    def get_plot_name(self):
        return "Random RBF Generator - {} class labels".format(self.num_classes)

    def get_classes(self):
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def generate_centroids(self):
        """ generate_centroids
        
        Sequentially creates all the centroids, choosing at random a center, 
        a label, a standard deviation and a weight. 
        
        """
        model_random = np.random
        model_random.seed(self.model_seed)
        self.centroids = []
        self.centroid_weights = []
        for i in range(self.num_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.num_numerical_attributes):
                rand_centre.append(model_random.rand())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = model_random.randint(self.num_classes)
            self.centroids[i].std_dev = model_random.rand()
            self.centroid_weights.append(model_random.rand())

    def get_info(self):
        return 'RandomRBFGenerator: model_seed: ' + str(self.model_seed) + \
               ' - instance_seed: ' + str(self.instance_seed) + \
               ' - num_classes: ' + str(self.num_classes) + \
               ' - num_att: ' + str(self.num_numerical_attributes) + \
               ' - num_centroids: ' + str(self.num_centroids)

    def get_num_targeting_tasks(self):
        return 1


class Centroid:
    """ Centroid
    
    Class that stores a centroid's attributes. No further methods.
    
    """
    def __init__(self):
        self.centre = None
        self.class_label = None
        self.std_dev = None
