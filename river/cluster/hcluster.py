from river import base, utils
import numpy as np
import numpy.typing as npt
from typing import Callable

# Node of a binary tree for Hierarchical Clustering
class BinaryTreeNode:
    def __init__(self, key : int, data : np.ndarray = None):
        # If it's a leaf : x data, else : None
        self.data = data
        # Key of the node
        self.key = key
        # Children and parent
        self.left = None
        self.right = None
        self.parent = None


def euclidean_distance(w_i, w_j):
        # Euclidean distance between two nodes
        return np.sqrt(np.sum(np.square(w_i.data - w_j.data)))

class HierarchicalClustering(base.Clusterer):
    """Hierarchical Clustering.

    HierarchicalClustering is a stream hierarchical clustering algorithm.
    The algorithm [^1] inserts new nodes near the nodes it is similar to, but without breaking clusters of very similar nodes.
    It will (begining with the whole tree T) :

    * Compare the new node to the tree T : 
        * if T is just a leaf : merge
        * else if the nodes of T are more similar between them than with the new node : merge
        * else if the new node is more similar to the left subtree than to the right subtree : redo from the first point with T = left subtree
        * else (the new node is more similar to the right subtree than to the left subtree) redo from the first point with T = right subtree
    
    You can choose a window size to use only the recent points and not overload the tree. (default : 100)

    You can also choose a distance function to compare the nodes. (default : euclidean distance)

    Parameters
    ----------
    window_size
        number of data points to use
    distance
        distance function to use to compare the nodes
    
    Attributes
    ----------
    n : int
        number of nodes
    X : dict (data point (str(np.ndarray)) : key (int))
        data points used by the algorithm with the key of the node representing them

    References
    ----------
    [^1]: Menon, Aditya & Rajagopalan, Anand & Sumengen, Baris & Citovsky, Gui & Cao, Qin & Kumar, Sanjiv. (2019). Online Hierarchical Clustering Approximations.

    Examples
    --------

    The first example is with leaving the window size to 100. In the second one we put it at 2 to see how it works.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [[1, 1, 1], [1, 1, 0], [20, 20, 20], [20, 20, 20.1], [0 , 1, 1]]

    >>> HC = cluster.HierarchicalClustering()

    >>> for x, _ in stream.iter_array(X):
    ...     HC = HC.learn_one(x)
    
    >>> HC.X
    {'[1 1 1]': 1,
    '[1 1 0]': 2,
    '[20 20 20]': 4,
    '[20.  20.  20.1]': 6,
    '[0 1 1]': 8}

    >>> HC.n
    9
    
    >>> print(HC)
            -> 6
        -> 7
            -> 4
    -> 5
            -> 8
        -> 9
                -> 2
            -> 3
                -> 1

    >>> HC.get_all_clusters()
    [(1, [array([1, 1, 1])]),
    (2, [array([1, 1, 0])]),
    (4, [array([20, 20, 20])]),
    (6, [array([20. , 20. , 20.1])]),
    (8, [array([0, 1, 1])]),
    (3, [1, 2]),
    (5, [9, 7]),
    (7, [4, 6]),
    (9, [3, 8])]

    >>> HC.get_clusters_by_point()
    {'[1 1 1]': [1, 3, 9, 5],
    '[1 1 0]': [2, 3, 9, 5],
    '[20 20 20]': [4, 7, 5],
    '[20.  20.  20.1]': [6, 7, 5],
    '[0 1 1]': [8, 9, 5]}

    >>> HC.predict_one({0 : 20.1, 1 : 20, 2 : 20 })
    ([10, 11, 5], 9)

    >>> HC = HC.learn_one({0 : 20.1, 1 : 20, 2 : 20 })

    >>> print(HC)
            -> 6
        -> 7
            -> 4
    -> 5
            -> 10
        -> 11
                -> 8
            -> 9
                    -> 2
                -> 3
                    -> 1


    >>> HC = cluster.HierarchicalClustering(window_size=2)

    >>> for x, _ in stream.iter_array(X):
    ...     HC = HC.learn_one(x)
    
    >>> HC.X
    {'[20.  20.  20.1]': 2, '[0 1 1]': 1}

    >>> HC.n
    3

    >>> print(HC)
        -> 2
    -> 3
        -> 1
    
    """
    def __init__(self, window_size : int = 100, distance : Callable[[BinaryTreeNode, BinaryTreeNode], float] = euclidean_distance):
        # Number of nodes
        self.n = 0
        # Max number of leaves
        self.w_size = window_size
        # Dict : x data (str(array of size m)) -> key of the node
        self.X : dict[np.ndarray, int] = {}
        # Dict : key -> node
        self.nodes : dict[int, BinaryTreeNode] = {}
        # First node of the tree
        self.root = None
        # Distance function
        self.distance = distance
    
    def OTD(self, T, x):
        # OTD algorithm from 
        if self.n == 1:
            # First node in the tree
            self.root = self.nodes[1]
        elif T.data is not None:
            # If T is a leaf, we merge the two nodes together
            self.merge_nodes(T, self.nodes[self.X[np.array2string(x)]])
        elif T.left is None:
            # If there is no node at the left of the intermediate node, we add it there
            T.left = self.nodes[self.X[np.array2string(x)]]
            self.nodes[self.X[np.array2string(x)]].parent = T
        elif T.right is None:
            # If there is no node at the right of the intermediate node, we add it there
            T.right = self.nodes[self.X[np.array2string(x)]]
            self.nodes[self.X[np.array2string(x)]].parent = T
        elif self.intra_subtree_similarity(T) < self.inter_subtree_similarity(T, self.nodes[self.X[np.array2string(x)]]):
            # If the nodes in T are closer between them than with the new node, we merge T and the new node
            self.merge_nodes(T, self.nodes[self.X[np.array2string(x)]])
        elif self.inter_subtree_similarity(T.left, self.nodes[self.X[np.array2string(x)]]) > self.inter_subtree_similarity(T.right, self.nodes[self.X[np.array2string(x)]]):
            # If the right part of the tree T is closer to the new node than the left part, we continue to search where to merge the new node in the right part of T
            self.OTD(T.right, x)
        else:
            # Else the left part of the tree T is closer to the new node than the right part, we continue to search where to merge the new node in the left part of T
            self.OTD(T.left, x)
    
    def merge_nodes(self, T, added_node):
        # Merge a new node (added node) to the tree T
        # We create the node that will be the parent of T and the added node
        self.n +=1
        new_node = BinaryTreeNode(self.n)
        # We add T and the added node as its children
        new_node.left = T
        new_node.right = added_node
        # The parent of the new node is the parent of T
        new_node.parent = T.parent
        # If T is not the root, we set the child of its parent as new node (instead of T)
        if T.parent is not None:
            if T.parent.left.key == T.key:
                T.parent.left = new_node
            else:
                T.parent.right = new_node
        # We add the new node as the parent of T and the added node
        T.parent = new_node
        added_node.parent = new_node
        # We add the new node to the dict
        self.nodes[self.n] = new_node
        # If T was the root, the new node become the root
        if self.root.key == T.key:
            self.root = self.nodes[self.n]
    
    def learn_one(self, x):
        x = utils.dict2numpy(x)
        # We create the node for x and add it to the tree
        if self.w_size > 0 and len(self.X.keys()) >= self.w_size:
            # If we have reached the maximum of data points, we delete the oldest one and add a node with the same key as the one we deleted
            oldest_key = self.X[list(self.X.keys())[0]]
            oldest = self.nodes[oldest_key]
            if oldest.parent.left.key == oldest_key:
                oldest.parent.left = None
            else:
                oldest.parent.right = None
            del self.nodes[oldest_key]
            del self.X[list(self.X.keys())[0]]
            self.X[np.array2string(x)] = oldest_key
            self.nodes[oldest_key] = BinaryTreeNode(oldest_key,x)
        else:
            # Else we just add a node
            self.n +=1
            self.X[np.array2string(x)] = self.n
            self.nodes[self.n] = BinaryTreeNode(self.n,x)
        # We add it to the tree
        self.OTD(self.root, x)
        return self
    
    def predict_OTD(self, x, node, clusters):
        # get the list of predicted clusters for x
        if node is None:
            # If there is still no node in the tree
            return [1], -1
        if node.data is not None:
            # If we compare x to a leaf, then we add itself (n+1) and the key of the node that would merge x and node (n+2)
            clusters.extend([self.n+2, self.n+1])
            return clusters, node.key
        if self.intra_subtree_similarity(node) < self.inter_subtree_similarity(node, BinaryTreeNode(self.n+1,x)):
            # If the nodes in the tree 'node' are closer betweem them than with x, then we add itself (n+1) and the key of the node that would merge x and node (n+2)
            clusters.extend([self.n+2, self.n+1])
            return clusters, node.key
        else:
            # Else, x wouls be added in the tree, we add the key of node
            clusters.extend([node.key])
            if self.inter_subtree_similarity(node.left,BinaryTreeNode(self.n+1,x)) > self.inter_subtree_similarity(node.right, BinaryTreeNode(self.n+1,x)):
                # If  the right part of the tree is closer to x than the left part, we continue in the right part
                return self.predict_OTD(x, node.right, clusters)
            else:
                # If  the left part of the tree is closer to x than the right part, we continue in the left part
                return self.predict_OTD(x, node.left, clusters)

    def predict_one(self, x):
        """Predicts the clusters for a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        Returns
        -------
        (list, int)
            A list of clusters (from node `x` to root) and the node to which it would have been merged.
        
        """
        x = utils.dict2numpy(x)
        # We predict to which cluster x would be if we added it in the tree
        r, merged = self.predict_OTD( x, self.root, [])
        r.reverse()
        return r, merged
    
    def find_path(self, root, path, k):
        # find the path from root to k, from https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/

        if root is None:
            # No path
            return False
    
        path.append(root)
    
        if root.key == k:
            return True
    
        if ((root.left is not None and self.find_path(root.left, path, k)) or
                (root.right is not None and self.find_path(root.right, path, k))):
            return True
    
        path.pop()
        return False

    def lca(self, i, j):
        # find the least common ancestor, from https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/
        if self.root is None:
            return -1
        
        path_i = []
        path_j = []
    
        if (not self.find_path(self.root, path_i, i) or not self.find_path(self.root, path_j, j)):
            return -1

        k = 0
        while(k < len(path_i) and k < len(path_j)):
            if path_i[k] != path_j[k]:
                break
            k += 1
        return path_i[k-1]
    
    def leaves(self, v):
        # find all the leaves from node v
        if v is None:
            # No leaves
            return -1
        if v.data is not None:
            # If v is a leaf, returns itself
            return [v]
        # Else, returns leaves from its children
        leavesList = []
        leavesList.extend(self.leaves(v.left))
        leavesList.extend(self.leaves(v.right))
        return leavesList

    def inter_subtree_similarity(self, A, B):
        # compute the mean distance between tree A and tree B (mean of distances between nodes from tree A and tree B)
        leavesA = self.leaves(A)
        leavesB = self.leaves(B)
        r = 0
        nb = 0
        for i, w_i in enumerate(leavesA):
            for j, w_j in enumerate(leavesB):
                nb +=1
                r += self.distance(w_i, w_j)
        return r/nb

    def intra_subtree_similarity(self, A):
        # compute mean of distances between the node from tree A
        leaves = self.leaves(A)
        r = 0
        nb = 0
        if len(leaves) == 1:
            return 0
        for i, w_i in enumerate(leaves):
            for j, w_j in enumerate(leaves):
                if i < j:
                    nb +=1
                    r += self.distance(w_i, w_j)
        return r/nb
    
    def __str__(self):
        self.printTree(self.root)
        return ""
    
    def printTree(self, node, level=0):
        # print node and its children, from https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        if node is not None:
            self.printTree(node.right, level + 1)
            print(' ' * 4 * level + '-> ' + str(node.key))
            self.printTree(node.left, level + 1)
    
    def get_parents(self,node):
        # Get all the parents of the node (the clusters it belongs to)
        clusters = [node.key]
        if node.parent is None:
            return clusters
        clusters.extend(self.get_parents(node.parent))
        return clusters
    
    def get_clusters_by_point(self):
        """Returns the list of clusters (from the data point node to the root) for all of the data points.

        Returns
        -------
        {x : list}
            A dict of all the data points with their clusters.
        """
        # Get all the clusters each data point belong to
        clusters = {}
        for x in self.X.keys():
            clusters[x] = self.get_parents(self.nodes[self.X[x]])
        return clusters
    
    def get_all_clusters(self):
        """Returns all the clusters of the tree.

        Returns
        -------
        {int : list}
            A dict of all the clusters with their children (or the data point for the leaves).
        """
        # Get the data of each cluster
        clusters = {}
        for i in range(1,self.n+1):
            if self.nodes[i].data is not None:
                clusters[i] = [self.nodes[i].data]
            else:
                clusters[i] = [self.nodes[i].left.key, self.nodes[i].right.key]
        return sorted(clusters.items(), key = lambda x : len(x[1]))
