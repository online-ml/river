import warnings
import numpy as np
from scipy import spatial
from skmultiflow.lazy import KDTree
from sklearn import neighbors as ng
from timeit import default_timer as timer
from skmultiflow.data import FileStream
from skmultiflow.transform import OneHotToCategorical


def demo():
    """ _test_kdtree_compare
    
    This demo compares creation and query speed for different kd tree 
    implementations. They are fed with instances from the covtype dataset. 
    
    Three kd tree implementations are compared: SciPy's KDTree, NumPy's 
    KDTree and scikit-multiflow's KDTree. For each of them the demo will 
    time the construction of the tree on 1000 instances, and then measure 
    the time to query 100 instances. The results are displayed in the 
    terminal.
    
    """
    warnings.filterwarnings("ignore", ".*Passing 1d.*")

    stream = FileStream('../data/datasets/covtype.csv', -1, 1)
    stream.prepare_for_use()
    filter = OneHotToCategorical([[10, 11, 12, 13],
                                  [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])

    X, y = stream.next_sample(1000)
    X = filter.transform(X)
    # print(X)

    X_find, y = stream.next_sample(100)
    X_find = filter.transform(X_find)
    print(X_find[4])
    # Normal kdtree
    start = timer()
    scipy = spatial.KDTree(X, leafsize=40)
    end = timer()
    print("\nScipy KDTree construction time: " + str(end-start))

    start = timer()
    for i in range(10):
        ind = scipy.query(X_find[i], 8)
        # print(ind)
    end = timer()
    print("Scipy KDTree query time: " + str(end - start))

    del scipy

    # Fast kdtree
    start = timer()
    opt = KDTree(X, metric='euclidean', return_distance=True)
    end = timer()
    print("\nOptimal KDTree construction time: " + str(end-start))

    start = timer()
    for i in range(100):
        ind, dist = opt.query(X_find[i], 8)
        # print(ind)
        # print(dist)
    end = timer()
    print("Optimal KDTree query time: " + str(end - start))

    del opt

    # Sklearn kdtree
    start = timer()
    sk = ng.KDTree(X, metric='euclidean')
    end = timer()
    print("\nSklearn KDTree construction time: " + str(end-start))

    start = timer()
    for i in range(100):
        ind, dist = sk.query(np.asarray(X_find[i]).reshape(1, -1), 8, return_distance=True)
        # print(ind)
        # print(dist)
    end = timer()
    print("Sklearn KDTree query time: " + str(end - start) + "\n")

    del sk


if __name__ == '__main__':
    demo()
