__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.base_object import BaseObject
from skmultiflow.classification.lazy.neighbours.distances import euclidean_distance


class KDTree(BaseObject):
    def __init__(self):
        super().__init__()
        self.dis_list = []
        self.inst_list = []
        self.root = None


    @property
    def _root(self):
        return self.root

    def get_info(self):
        return 'Not implemented.'


class KDTreeNode(BaseObject):
    def __init__(self, node_num, start_index, end_index, node_range, rect_boundaries):
        super().__init__()
        self.node_number = node_num
        self.left_subtree = None
        self.right_subtree = None
        self.split_val = None
        self.split_index = None
        self.node_range = [] if node_range is None else node_range
        self.hyper_rect_boundaries = [] if rect_boundaries is None else rect_boundaries
        self.start = start_index
        self.end = end_index

    def get_info(self):
        return 'Not implemented.'

    def get_class_type(self):
        return 'data_structure'