__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.instances.InstanceHeader import InstanceHeader
from skmultiflow.core.instances.InstanceData import InstanceData
from skmultiflow.data.BaseInstanceStream import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
from skmultiflow.core.BaseObject import BaseObject
import numpy as np
from array import array
'''
    keep track of the last instance generated
'''


class RandomTreeGenerator(BaseInstanceStream, BaseObject):
    '''
        RandomTreeGenerator
        ---------------------------------------------
        Instance generator based on a random tree that splits attributes and sets labels to its leafs.
        Supports the addition of noise.
        
        Parser parameters
        ---------------------------------------------
        -r: Seed for random generation of tree (Default: 23)
        -i: Seed for random generation of instances (Default: 12)
        -c: The number of classes to generate (Default: 2)
        -o: The number of nominal attributes to generate (Default: 5)
        -u: The number of numerical attributes to generate (Default: 5)
        -v: The number of values to generate per nominal attribute (Default: 5)
        -d: The maximum depth of the tree concept (Default: 5)
        -l: The first level of the tree above MaxTreeDepth that can have leaves (Default: 3)
        -f: The fraction of leaves per level from FirstLeafLevel onwards (Default: 0.15)
        
        Instance Data Format
        --------------------------------------------
        The data coming from this generator follows a strict form:
         ___________________________________________________________
        |   Numeric Attributes  |   Nominal Attributes  |   Class   |
         -----------------------------------------------------------
    '''
    def __init__(self, opt_list = None):
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
        self.configure(opt_list)
        pass

    def configure(self, opt_list):
        if opt_list is not None:
            for i in range(len(opt_list)):
                opt, arg = opt_list[i]
                if opt in ("-r"):
                    self.random_tree_seed = int(arg)
                elif opt in ("-i"):
                    self.random_instance_seed = int(arg)
                elif opt in ("-c"):
                    self.num_classes = int(arg)
                elif opt in ("-o"):
                    self.num_nominal_attributes = int(arg)
                elif opt in ("-u"):
                    self.num_numerical_attributes = int(arg)
                elif opt in ("-v"):
                    self.num_values_per_nominal_att = int(arg)
                elif opt in ("-d"):
                    self.max_tree_depth = int(arg)
                elif opt in ("-l"):
                    self.min_leaf_depth = int(arg)
                elif opt in ("-f"):
                    self.fraction_of_leaves_per_level = float(arg)
        self.class_header = InstanceHeader(["class"])
        self.attributes_header = []
        for i in range(self.num_numerical_attributes):
            self.attributes_header.append("NumAtt" + str(i))
        for i in range(self.num_nominal_attributes):
            for j in range(self.num_values_per_nominal_att):
                self.attributes_header.append("NomAtt" + str(i) + "_Val" + str(j))

        #print(self.attributes_header)

        pass

    def prepare_for_use(self):
        self.instance_random = np.random
        self.instance_random.seed(self.random_instance_seed)
        self.restart()

    def generate_random_tree(self):
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
        if len(node.children) == 0:
            return node.class_label
        if node.split_att_index < self.num_numerical_attributes:
            aux = 0 if att_vals[node.split_att_index] < node.split_att_value else 1
            return self.classify_instance(node.children[aux], att_vals)
        else:
            return self.classify_instance(node.children[self.get_integer_nominal_attribute_representation(node.split_att_index, att_vals)], att_vals)

    def get_integer_nominal_attribute_representation(self, nominal_index = None, att_vals = None):
        '''
            The nominal_index uses as reference the number of nominal attributes plus the number of numerical attributes.
             In this way, to find which 'hot one' variable from a nominal attribute is active, we do some basic math.
             This function returns the index of the active variable in a nominal attribute 'hot one' representation.
        '''
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
        #att = array('d')
        num_attributes = -1
        data = np.zeros([batch_size, self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att) + 1])
        for j in range (batch_size):
            for i in range(self.num_numerical_attributes):
                data[j,i] = self.instance_random.rand()

            for i in range(self.num_numerical_attributes, self.num_numerical_attributes+(self.num_nominal_attributes*self.num_values_per_nominal_att), self.num_values_per_nominal_att):
                aux = self.instance_random.randint(0, self.num_values_per_nominal_att)
                for k in range(self.num_values_per_nominal_att):
                    if aux == k:
                        data[j, k+i] = 1.0
                    else:
                        data[j, k+i] = 0.0

            data[j, self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)] = self.classify_instance(self.tree_root, data[j])
            #att.append(self.classify_instance(self.treeRoot, att))
            self.current_instance_x = data[:self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)]
            self.current_instance_y = data[self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att):]
            #self.current_instance = Instance(self.num_nominal_attributes*self.num_values_per_nominal_att + self.num_numerical_attributes, self.num_classes, -1, att)
            num_attributes = self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)
        return (data[:, :num_attributes], np.ravel(data[:, num_attributes:]))

    def is_restartable(self):
        return True

    def restart(self):
        self.instance_index = 0
        self.instance_random = np.random
        self.instance_random.seed(self.random_instance_seed)
        self.generate_random_tree()
        pass

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

    def get_num_classes(self):
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
        pass

class Node:
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