__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.lazy.neighbors.kdtree import KDTree


def demo():
    kdtree = KDTree([[0.543,4,1,6,9,7,3,6],[1237,4,6,8,3,7,3,6],[0.035,356,3,7,3,2,88,2],[235,6,3,86,21,65,1,5]],
                    metric='euclidean', categorical_list=[[1,2], [5,6,7]])
    print(kdtree.n_features)


if __name__ == '__main__':
    demo()