__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.instances.instance_header import InstanceHeader
from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.base_object import BaseObject
import numpy as np
from array import array


class RandomTreeGenerator(BaseInstanceStream, BaseObject):
    """ RandomTreeGenerator
       
    This generator is built based on its description in Domingo and Hulten's 
    'Knowledge Discovery and Data Mining'. The generator is based on a random 
    tree that splits attributes at random and sets labels to its leafs.
    
    The tree structure is composed on Node objects, which can be either inner 
    nodes or leaf nodes. The choice comes as a function fo the parameters 
    passed to its initializer.
    
    Since the concepts are generated and classified according to a tree 
    structure, in theory, it should favour decision tree learners.
    
    Parameters
    ----------
    tree_seed: int (Default: 23)
        Seed for random generation of tree.
    
    instance_seed: int (Default: 12)
        Seed for random generation of instances.
    
    n_classes: int (Default: 2)
        The number of targets to generate.
    
    n_nominal_attributes: int (Default: 5)
        The number of nominal attributes to generate.
    
    n_numerical_attributes: int (Default: 5)
        The number of numerical attributes to generate.
    
    n_values_per_nominal: int (Default: 5)
        The number of values to generate per nominal attribute.
    
    max_depth: int (Default: 5)
        The maximum depth of the tree concept.
    
    min_leaf_depth: int (Default: 3)
        The first level of the tree above MaxTreeDepth that can have leaves.
    
    fraction_leaves_per_level: float (Default: 0.15)
        The fraction of leaves per level from min_leaf_depth onwards.
    
    """
    def __init__(self, tree_seed=23, instance_seed=12, n_classes=2, n_nominal_attributes=5,
                 n_numerical_attributes=5, n_values_per_nominal=5, max_depth=5, min_leaf_depth=3,
                 fraction_leaves_per_level=0.15):
        super().__init__()

        # default values
        self.random_tree_seed = 23
        self.random_instance_seed = 12
        self.num_classes = 2
        self.num_numerical_attributes = 5
        self.num_nominal_attributes = 5
        self.num_values_per_nominal_att = 5
        self.max_tree_depth = 5
        self.min_leaf_depth = 3
        self.fraction_of_leaves_per_level = 0.15
        self.instance_index = 0
        self.tree_root = None
        self.instance_random = None
        self.class_header = None
        self.attributes_header = None
        self.instance_random = None
        self.current_instance_x = None
        self.current_instance_y = None
        self.__configure(tree_seed, instance_seed, n_classes, n_nominal_attributes,
                         n_numerical_attributes, n_values_per_nominal, max_depth, min_leaf_depth,
                         fraction_leaves_per_level)

    def __configure(self, tree_seed, instance_seed, n_classes, n_nominal_attributes,
                    n_numerical_attributes, n_values_per_nominal, max_depth, min_leaf_depth,
                    fraction_leaves_per_level):
        self.random_tree_seed = tree_seed
        self.random_instance_seed = instance_seed
        self.num_classes = n_classes
        self.num_nominal_attributes = n_nominal_attributes
        self.num_numerical_attributes = n_numerical_attributes
        self.num_values_per_nominal_att = n_values_per_nominal
        self.max_tree_depth = max_depth
        self.min_leaf_depth = min_leaf_depth
        self.fraction_of_leaves_per_level = fraction_leaves_per_level

        self.class_header = InstanceHeader(["class"])
        self.attributes_header = []
        for i in range(self.num_numerical_attributes):
            self.attributes_header.append("NumAtt" + str(i))
        for i in range(self.num_nominal_attributes):
            for j in range(self.num_values_per_nominal_att):
                self.attributes_header.append("NomAtt" + str(i) + "_Val" + str(j))

    def prepare_for_use(self):
        self.instance_random = np.random
        self.instance_random.seed(self.random_instance_seed)
        self.restart()

    def generate_random_tree(self):
        """ generate_random_tree
        
        Generates the random tree, starting from the root node and following 
        the constraints passed as parameters to the initializer. 
        
        The tree is recursively generated, node by node, until it reaches the
        maximum tree depth.
        
        """
        # Starting random generators and parameter arrays
        tree_rand = np.random
        tree_rand.seed(self.random_tree_seed)
        nominal_att_candidates = array('i')
        min_numeric_vals = array('d')
        max_numeric_vals = array('d')

        for i in range(self.num_numerical_attributes):
            min_numeric_vals.append(0.0)
            max_numeric_vals.append(1.0)

        for i in range(self.num_numerical_attributes + self.num_nominal_attributes):
            nominal_att_candidates.append(i)

        self.tree_root = self.generate_random_tree_node(0, nominal_att_candidates, min_numeric_vals, max_numeric_vals, tree_rand)


    def generate_random_tree_node(self, current_depth, nominal_att_candidates, min_numeric_vals, max_numeric_vals, rand):
        """ generate_random_tree_node
        
        Creates a node, choosing at random the splitting attribute and the 
        split value. Fill the features with random feature values, and then 
        recursively generates its children. If the split attribute is a 
        numerical attribute there are going to be two children nodes, one
        for samples where the value for the split attribute is smaller than 
        the split value, and one for the other case.
        
        Once the recursion passes the leaf minimum depth, it probabilistic 
        chooses if the node is a leaf or not. If not, the recursion follow 
        the same way as before. If it decides the node is a leaf, a class 
        label is chosen for the leaf at random.
        
        Furthermore, if the current_depth is equal or higher than the tree 
        maximum depth, a leaf node is immediately returned.
        
        Parameters
        ----------
        current_depth: int
            The current tree depth.
        
        nominal_att_candidates: list
            A list containing all the, still not chosen for the split, 
            nominal attributes.
        
        min_numeric_vals: list 
            The minimum value reachable, at this branch of the 
            tree, for all numeric attributes.
        
        max_numeric_vals: list
            The minimum value reachable, at this branch of the 
            tree, for all numeric attributes.
            
        rand: numpy.random
            A numpy random generator instance.
        
        Returns
        -------
        Returns the node, either a inner node or a leaf node.
        
        Notes
        -----
        If the splitting attribute of a node happens to be a nominal attribute 
        we guarantee that none of its children will split on the same attribute, 
        as it would have no use for that split.
         
        """
        if ((current_depth >= self.max_tree_depth) | ((current_depth >= self.min_leaf_depth) & (self.fraction_of_leaves_per_level >= (1.0 - rand.rand())))):
            leaf = Node()
            leaf.class_label = rand.randint(0, self.num_classes)
            return leaf

        node = Node()
        chosen_att = rand.randint(0, len(nominal_att_candidates))
        if (chosen_att < self.num_numerical_attributes):
            numeric_index = chosen_att
            node.split_att_index = numeric_index
            min_val = min_numeric_vals[numeric_index]
            max_val = max_numeric_vals[numeric_index]
            node.split_att_value = ((max_val - min_val) * rand.rand() + min_val)
            node.children = []

            new_max_vals = max_numeric_vals[:]
            new_max_vals[numeric_index] = node.split_att_value
            node.children.append(self.generate_random_tree_node(current_depth + 1, nominal_att_candidates, min_numeric_vals, new_max_vals, rand))

            new_min_vals = min_numeric_vals[:]
            new_min_vals[numeric_index] = node.split_att_value
            node.children.append(self.generate_random_tree_node(current_depth + 1, nominal_att_candidates, new_min_vals, max_numeric_vals, rand))
        else:
            node.split_att_index = nominal_att_candidates[chosen_att]
            new_nominal_candidates = array('d', nominal_att_candidates)
            new_nominal_candidates.remove(node.split_att_index)

            for i in range(self.num_values_per_nominal_att):
                node.children.append(self.generate_random_tree_node(current_depth + 1, new_nominal_candidates, min_numeric_vals, max_numeric_vals, rand))

        return node

    def classify_instance(self, node, att_vals):
        """ classify_instance
        
        After a sample is generated it passes through this function, which 
        advances the tree structure until it finds a leaf node.
        
        Parameters
        ----------
        node: Node object
            The Node that will be verified. Either it's a leaf, and then the 
            label is returned, or it's a inner node, and so the algorithm 
            will continue to advance in the structure.
            
        att_vals: numpy.array
            The set of generated feature values of the sample.
        
        Returns
        -------
        Return a tuple with the features matrix and the labels matrix 
        for the batch_size samples that were requested.
        
        """
        if len(node.children) == 0:
            return node.class_label
        if node.split_att_index < self.num_numerical_attributes:
            aux = 0 if att_vals[node.split_att_index] < node.split_att_value else 1
            return self.classify_instance(node.children[aux], att_vals)
        else:
            return self.classify_instance(node.children[self.__get_integer_nominal_attribute_representation(node.split_att_index, att_vals)], att_vals)

    def __get_integer_nominal_attribute_representation(self, nominal_index = None, att_vals = None):
        """ __get_integer_nominal_attribute_representation
        
        Utility function, to determine a nominal index when coded in one-hot 
        fashion.
        
        The nominal_index uses as reference the number of nominal attributes 
        plus the number of numerical attributes. 
        
        Parameters
        ----------
        nominal_index: int
            The nominal feature index.
            
        att_vals: np.array
            The features array.
            
        Returns
        -------
        This function returns the index of the active variable in a nominal 
        attribute 'hot one' representation.
        
        """
        minIndex = self.num_numerical_attributes + (nominal_index - self.num_numerical_attributes) * self.num_values_per_nominal_att
        for i in range(self.num_values_per_nominal_att):
            if att_vals[int(minIndex)] == 1:
                return i
            minIndex += 1
        return None

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size = 1):
        """ next_instance
        
        Randomly generates attributes values, and then classify each instance 
        generated.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
         
        Returns
        -------
        Return a tuple with the features matrix and the labels matrix for the 
        batch_size samples that were requested.
         
        """
        num_attributes = -1
        data = np.zeros([batch_size, self.num_numerical_attributes + (self.num_nominal_attributes
                                                                      * self.num_values_per_nominal_att) + 1])
        for j in range (batch_size):
            for i in range(self.num_numerical_attributes):
                data[j,i] = self.instance_random.rand()

            for i in range(self.num_numerical_attributes, self.num_numerical_attributes+
                    (self.num_nominal_attributes*self.num_values_per_nominal_att), self.num_values_per_nominal_att):
                aux = self.instance_random.randint(0, self.num_values_per_nominal_att)
                for k in range(self.num_values_per_nominal_att):
                    if aux == k:
                        data[j, k+i] = 1.0
                    else:
                        data[j, k+i] = 0.0

            data[j, self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)] \
                = self.classify_instance(self.tree_root, data[j])

            self.current_instance_x = data[:self.num_numerical_attributes + (self.num_nominal_attributes
                                                                             * self.num_values_per_nominal_att)]

            self.current_instance_y = data[self.num_numerical_attributes + (self.num_nominal_attributes
                                                                            * self.num_values_per_nominal_att):]

            num_attributes = self.num_numerical_attributes + (self.num_nominal_attributes
                                                              * self.num_values_per_nominal_att)

        return (data[:, :num_attributes], np.ravel(data[:, num_attributes:]))

    def is_restartable(self):
        return True

    def restart(self):
        self.instance_index = 0
        self.instance_random = np.random
        self.instance_random.seed(self.random_instance_seed)
        self.generate_random_tree()

    def has_more_mini_batch(self):
        pass

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.num_numerical_attributes + self.num_nominal_attributes * self.num_values_per_nominal_att

    def get_num_targets(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.class_header

    def get_last_instance(self):
        return (self.current_instance_x, self.current_instance_y)

    def get_plot_name(self):
        return "Random Tree Generator - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def get_info(self):
        return 'RandomTreegenerator: tree_seed: ' + str(self.random_tree_seed) + \
               ' - instance_seed: ' + str(self.random_instance_seed) + \
               ' - n_classes: ' + str(self.num_classes) + \
               ' - n_nominal_attributes: ' + str(self.num_nominal_attributes) + \
               ' - n_numerical_attributes: ' + str(self.num_numerical_attributes) + \
               ' - n_values_per_nominal_attribute: ' + str(self.num_values_per_nominal_att) + \
               ' - max_depth: ' + str(self.max_tree_depth) + \
               ' - min_leaf_depth: ' + str(self.min_leaf_depth) + \
               ' - fraction_leaves_per_level: ' + str(self.fraction_of_leaves_per_level)

    def get_num_targeting_tasks(self):
        return 1

class Node:
    """ Node
    
    Class that stores the attributes of a node. No further methods.
    
    Parameters
    ----------
    class_label: int, optional
        If given it means the node is a leaf and the class label associated 
        with it is class_label.
        
    split_att_index: int, optional
        If given it means the node is an inner node and the split attribute 
        is split_att_index.
        
    split_att_value: int, optional
        If given it means the node is an inner node and the split value is 
        split_att_value.
    
    """
    def __init__(self, class_label = None, split_att_index = None, split_att_value = None):
        self.class_label = class_label
        self.split_att_index = split_att_index
        self.split_att_value = split_att_value
        self.children = []

if __name__ == "__main__":
    stream = RandomTreeGenerator()
    stream.prepare_for_use()

    for i in range(4):
        X, y = stream.next_instance(2)
        print(X)
        print(y)