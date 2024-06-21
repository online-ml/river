from __future__ import annotations
import typing

import math

from river import base

class HierarchicalHeavyHitters(base.Base):

    """Full Ancestry Algorithm implementation for the Hierarchical Heavy Hitters problem.[^1]

    The Hierarchical Heavy Hitters problem involves identifying the most frequent items in a data stream while organizing them hierarchically. 
    The Full Ancestry Algorithm leverages the hierarchical structure of the data to provide accurate frequency estimates by taking into account the frequencies of ancestor nodes.
    The algorithm operates in three principal phases:
    - **Insertion:** For every new data element received, the algorithm recursively tries to find the element in the trie. If it is present, it increments the counter of the element by its weight. Otherwise, its parent is recursively called until finding the closest one, or the root is reached.
    - **Compression:** After every `w` updates, the compression phase is triggered to reduce space usage by merging nodes with counts below the current bucket threshold. It merges nodes where the sum of their exact count and estimated error is less than or equal to the current bucket number minus one.
    - **Output:** This function generates a list of heavy hitters with frequency estimates above a threshold given by phi*N. It transverses the hierarchical tree and aggregates the frequencies of nodes that meet the specified criteria, ensuring that the output reflects the most significant elements in the data stream.

    Parameters
    ----------
    k
        The number of heavy hitters to track.
    epsilon
        The error parameter. Smaller values increase the accuracy but also the memory usage. Should be in $[0, 1]$.
    parent_func
        Function to fetch the parent of order i from child x. The function should return the root_value when i has reached the end of the tree and x when i equals 0. If this parameter is not given it defaults to a function that returns the prefix of length i of the input element.
    root_value:
        The value of the root node in the hierarchical tree. This parameter defines the starting point of the hierarchy. If no root value is specified, the root will be initialized when the first data element is processed and will have the value of None.

    Attributes
    ----------
    bucketSize : int
        The size of buckets used to compress counts.
    N : int
        The total number of updates processed.
    root : HierarchicalHeavyHitters.Node
        The root node of the hierarchical tree.

    Examples
    --------

    >>> from river import sketch

    >>> def custom_parent_func(x, i): 
    ...     if i < len(x):
    ...         return None  #root value
    ...     return x[:i]

    >>> hierarchical_hh = sketch.HierarchicalHeavyHitters(k=10, epsilon=0.001, parent_func=custom_parent_func, root_value=None)

    >>> for line in [1,2,21,31,34,212,3,24]:
    ...     hierarchical_hh.update(str(line))

    >>> print(hierarchical_hh)
    ge: 0, delta_e: 0, max_e: 0
    : 
        ge: 0, delta_e: 0, max_e: 0
        1: 
            ge: 1, delta_e: 0, max_e: 0
        2: 
            ge: 1, delta_e: 0, max_e: 0
            21: 
                ge: 1, delta_e: 0, max_e: 0
                212: 
                    ge: 1, delta_e: 0, max_e: 0
            24: 
                ge: 1, delta_e: 0, max_e: 0
        3: 
            ge: 1, delta_e: 0, max_e: 0
            31: 
                ge: 1, delta_e: 0, max_e: 0
            34: 
                ge: 1, delta_e: 0, max_e: 0

    >>> print( hierarchical_hh['212'])
    1

    >>> phi = 0.01
    >>> heavy_hitters = hierarchical_hh.output(phi)
    >>> print(heavy_hitters)
    [('1', 1), ('212', 1), ('21', 2), ('24', 1), ('2', 4), ('31', 1), ('34', 1), ('3', 3)]

    >>> def custom_parent_func2(x, i): 
    ...     parts = x.split('.')
    ...     if i >= len(parts):
    ...         return None  
    ...     return '.'.join(parts[:i+1])

    >>> hierarchical_hh = sketch.HierarchicalHeavyHitters(k=10, epsilon=0.001, parent_func=custom_parent_func2, root_value=None)

    >>> for line in ["123.456","123.123", "456.123", "123", "123"]:
    ...     hierarchical_hh.update(line)

    >>> print(hierarchical_hh)
    ge: 0, delta_e: 0, max_e: 0
    123: 
        ge: 2, delta_e: 0, max_e: 0
        123.456: 
            ge: 1, delta_e: 0, max_e: 0
        123.123: 
            ge: 1, delta_e: 0, max_e: 0
    456: 
        ge: 0, delta_e: 0, max_e: 0
        456.123: 
            ge: 1, delta_e: 0, max_e: 0
    
    >>> heavy_hitters = hierarchical_hh.output(phi)
    [('123.456', 1), ('123.123', 1), ('123', 4), ('456.123', 1)]

    References
    ----------
    - [^1]: Cormode, Graham, Flip Korn, S. Muthukrishnan, and Divesh Srivastava. "Finding hierarchical heavy hitters in streaming data." Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2010.
    """

    class Node:
        """Represents a node in the hierarchical tree structure used by HHH."""

    class Node:
        def __init__(self):
            self.ge = 0
            self.delta_e = 0
            self.max_e = 0
            self.fe = 0
            self.Fe = 0
            self.children: typing.Dict[typing.Hashable, HierarchicalHeavyHitters.Node] = {}

    def __init__(self, k: int, epsilon: float, parent_func: typing.Callable[[typing.Hashable, int], typing.Hashable] = None, root_value: typing.Hashable = None):
        self.k = k
        self.epsilon = epsilon
        self.bucketSize = math.floor(1 / epsilon)
        self.N = 0
        self.root = None if root_value is None else HierarchicalHeavyHitters.Node()
        self.parent_func = parent_func if parent_func is not None else lambda x, i: None if i > len(str(x)) else str(x)[:i]
        self.root_value = root_value

    def update(self, x: typing.Hashable, w: int = 1):
        """Update the count for a given hierarchical key with an optional weight."""

        self.N += 1
        if self.root is None:
            self.root = HierarchicalHeavyHitters.Node()
            self.root.delta_e = self.currentBucket() - 1
            self.root.max_e = self.root.delta_e
        
        current = self.root
        parent_me = 0

        sub_key = x
        i = 0

        while str(sub_key)!=str(self.root_value):
     
            sub_key = self.parent_func(x, i)
         
            i+=1
         
            if str(sub_key) == str(self.root_value):
                if self.root is None:
                    self.root = HierarchicalHeavyHitters.Node()
                    self.root.delta_e = self.currentBucket() - 1
                    self.root.max_e = self.root.delta_e
                current = self.root

            
            elif sub_key in current.children:
                current = current.children[sub_key]
                if str(sub_key) == str(x):
                    current.ge += w

            else:
                current.children[sub_key] = HierarchicalHeavyHitters.Node()
                current = current.children[sub_key]
                current.delta_e = parent_me
                current.max_e = parent_me

                if str(sub_key) == str(x):
                    current.ge += w

            parent_me = current.max_e

        self.compress()

    def currentBucket(self):
        """Calculate the current bucket number based on the total updates processed."""
        return math.ceil(self.N / self.bucketSize)

    def compress(self):
        """Compress the hierarchical tree by merging nodes with counts below the current bucket threshold."""
        if (self.N % self.bucketSize == 0):
            self._compress_node(self.root)
        
    def _compress_node(self, node: HierarchicalHeavyHitters.Node):
        """Recursively compress nodes in the hierarchical tree."""
        if not node.children:
            return

        for child_key, child_node in list(node.children.items()):

            if not child_node.children=={} :
                self._compress_node(child_node)
            
            else:
          
                if child_node.ge + child_node.delta_e <= self.currentBucket() - 1:
                    node.ge += child_node.ge
                    node.max_e = max (node.max_e, child_node.ge + child_node.delta_e)
                    del node.children[child_key]

            
    def output(self, phi: float) -> list[typing.Hashable]:
        """Generate a list of heavy hitters with frequency estimates above the given threshold."""
        result = []
        self.root.fe = 0
        self.root.Fe = 0

        for _, child_node in list(self.root.children.items()):
            child_node.fe = 0
            child_node.Fe = 0

        self._output_node(self.root, phi, result)
        return result

    def _output_node(self, node: HierarchicalHeavyHitters.Node, phi: float, result: list):
        """Recursively generate heavy hitters from the hierarchical tree."""
        if not node.children:
            return
        
        for child_key, child_node in list(node.children.items()):

            if not child_node.children=={} :
                self._output_node(child_node, phi, result)

            if child_node.ge + node.ge + node.delta_e >= phi * self.N:
                result.append((child_key,child_node.fe + child_node.ge + child_node.delta_e))

            else:
                node.Fe += child_node.Fe + child_node.ge

            node.fe += child_node.fe + child_node.ge
            
    def __getitem__(self, key: typing.Hashable) -> int:
        """Get the count of a specific hierarchical key."""
        current = self.root

        for i in range(len(key)):
                
                sub_key = key[:i + 1]


                if sub_key not in current.children:

                    return 0
                
                current = current.children[sub_key]

                if sub_key == key:
               
                   return current.ge
 
            
    def totals(self) -> int:
        """Return the total number of elements in the hierarchical tree."""
        return self._count_entries(self.root) -1
    
    def _count_entries(self, node: HierarchicalHeavyHitters.Node) -> int:
        """Recursively count the total number of nodes in the hierarchical tree."""
        total = 1  
        
        for child_node in node.children.values():
            total += self._count_entries(child_node)
        
        return total
    
    def __str__(self):
        """Return a string representation of the hierarchical tree."""
        if self.root == None:
            return "None"
        return self._print_node(self.root, 0)

    def _print_node(self, node: HierarchicalHeavyHitters.Node, level: int) -> str:
        """Recursively generate a string representation of the hierarchical tree."""
        indent = ' ' * 4
        result = ''
        result += f"{indent * level}ge: {node.ge}, delta_e: {node.delta_e}, max_e: {node.max_e}\n"
        for child_key, child_node in node.children.items():
            result += f"{indent * level}{child_key }: \n"
            result += self._print_node(child_node, level + 1)
        return result
    
  
