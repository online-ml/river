__author__ = 'Guilherme Matsumoto'

import numpy as np
import math
from sklearn.utils import tosequence
import time


def dict_to_list(dict):
    """ dict_to_list
    
    Creates a list, based on the entries of a dictionary. To do so it ignores the 
    dictionary's keys, accounting only for its indexes.
    
    Parameters
    ----------
    dict: dictionary
        The dictionary that will originate the list.
     
    Returns
    -------
    list
        The generated list, where the first dimension is relative to the dictionary's 
        key indexes and the second dimension represents the dictionary's values.
    
    """
    aux = []
    keys = list(dict.keys())
    vals = list(dict.values())
    for i in range(len(dict)):
        aux.append([keys[i], vals[i]])
    return aux


def tuple_list_to_list(tup_list):
    """ tuple_list_to_list 
    
    Generates a list based on a list of tuples.
    
    Parameters
    ----------
    tup_list: list of tuples.
        The tuple list to be converted to a list.
    
    Returns
    -------
    list
        A list, where each entry contains two elements, which are the corresponding 
        tuple's entries.
    
    """
    aux = []
    for i in range(len(tup_list)):
        aux.append([tup_list[i][0], tup_list[i][1]])
    return aux


def dict_to_tuple_list(dict):
    """ dict_to_tuple_list
    
    Generates a tuple list based on a dict. It does so, by converting each 
    of the dictionary's entry into two entries of the same list's index and then 
    converting each of the lists entries into a tuple.
    
    Parameters
    ----------
    dict: dictionary.
        The dictionary to be converted to a tuple list.
    
    Returns
    -------
    list
        A list, where each entry contains two elements, which are the corresponding 
        tuple's entries.
    
    """
    aux = []
    list = dict_to_list(dict)
    for i in range(len(list)):
        aux.append((list[i][0], list[i][1]))
    return aux


def get_dimensions(X):
    """ get_dimensions
    
    Returns the dimensions from a numpy.array, numpy.ndarray or list.
    
    
    Parameters
    ----------
    X: numpy.array, numpy.ndarray, list, list of lists.
    
    Returns
    -------
    tuple
        A tuple representing the X structure's dimensions.
    
    """
    r, c = 1, 1
    if isinstance(X, type(np.array([0]))):
        if X.ndim > 1:
            r, c = X.shape
        else:
            r, c = 1, X.size

    elif isinstance(X, type([])):
        if isinstance(X[0], type([])):
            r, c = len(X), len(X[0])
        else:
            c = len(X)

    return r, c


def normalize_values_in_dict(dictionary, factor=None):
    """ Normalize the values in a dictionary using the given factor.

    Parameters
    ----------
    dictionary: dict
        Dictionary to normalize.
    factor: float, optional (default=None)
        Normalization factor value. If not set, use the sum of values.


    """
    if factor is None:
        factor = sum(dictionary.values())
    if factor == 0:
        raise ValueError('Can not normalize, normalization factor is zero')
    if math.isnan(factor):
        raise ValueError('Can not normalize, normalization factor is NaN')
    for key, value in dictionary.items():  # loop over the keys, values in the dictionary
        dictionary[key] = value / factor


def get_max_value_index(dictionary):
    """ Find the index of the maximum value in a dictionary.

    Parameters
    ----------
    dictionary: dict
        Dictionary to evaluate.

    Returns
    -------
    int
        Index of the maximum value.
    """
    if dictionary and isinstance(dictionary, dict):
        return max(dictionary, key=dictionary.get)
    else:
        return 0


if __name__ == '__main__':
    aux1 = {'key1': 1.0, 'key2': 2.3, 'key3': 3.5, 'key4': 4.8}
    aux = [('key1', 1.0), ('key2', 2.3), ('key3', 3.5), ('key4', 4.8)]
    #list1 = dict_to_list(aux1)
    #list = tuple_list_to_list(aux)
    #list_to_sequence = tosequence(aux)
    #print(list)
    #print(list1)

    #tuple_list = dict_to_tuple_list(aux1)

    test = tosequence(aux)
    print(test)
    print(test[-1][-1])
    time.sleep(20)

    #print(tuple_list)