import math
import numpy as np


def mixed_distance(instance_one, instance_two, **kwargs):
    n_att1 = instance_one.size if hasattr(instance_one, 'size') else len(instance_one)
    n_att2 = instance_two.size if hasattr(instance_two, 'size') else len(instance_two)

    if n_att1 != n_att2:
        if 'index' in kwargs and n_att2 == 1:
            instance_two = [instance_two] * n_att1
        else:
            raise ValueError("The two instances must have the same length.")

    index = -1
    categorical_list = None
    distance_array = None
    for key, value in kwargs.items():
        if key == 'categorical_list':
            categorical_list = value
        if key == 'index':
            index = value
        if key == 'distance_array':
            distance_array = value

    if index != -1:
        if categorical_list is not None:
            if len(categorical_list) > 0:
                if index in categorical_list:
                    if instance_one[index] != instance_two[index]:
                        return 1
                    else:
                        return 0
                else:
                    if distance_array is not None:
                        return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))\
                               / (distance_array[index]*distance_array[index]) if distance_array[index] != 0 else 0.0
                    else:
                        return (instance_one[index]-instance_two[index]) * (instance_one[index]-instance_two[index])
            else:
                if distance_array is not None:
                    return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))\
                               / (distance_array[index]*distance_array[index]) if distance_array[index] != 0 else 0.0
                else:
                    return (instance_one[index] - instance_two[index]) * (instance_one[index] - instance_two[index])
        else:
            if distance_array is not None:
                return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))\
                               / (distance_array[index]*distance_array[index]) if distance_array[index] != 0 else 0.0
            else:
                return (instance_one[index]-instance_two[index]) * (instance_one[index]-instance_two[index])

    partial_dist = []
    for i in range(n_att1):
        if categorical_list is not None:
            if i in categorical_list:
                if instance_one[i] != instance_two[i]:
                    partial_dist.append(1)
                else:
                    partial_dist.append(0)
            else:
                if not distance_array[i] == 0:
                    partial_dist.append(math.pow(instance_one[i] - instance_two[i], 2)/(distance_array[i]*distance_array[i]))
                else:
                    partial_dist.append(0.0)
        else:
            if not distance_array[i] == 0:
                partial_dist.append(
                    math.pow(instance_one[i] - instance_two[i], 2) / (distance_array[i] * distance_array[i]))
            else:
                partial_dist.append(0.0)

    return sum(partial_dist)


def euclidean_distance(instance_one, instance_two, **kwargs):
    """ Euclidean distance
    
    Function to compute the euclidean distance between two instances. 
    When measuring distances between instances, it's important to know 
    the properties of the samples' features. It won't work well with a 
    mix of categorical and numerical features.
    
    Parameters
    ----------
    instance_one: array-like of size n_features
        An array representing the features of a sample.
        
    instance_two: array-like of size n_features <possible a numeric value>
        An array representing the features of a sample.
        
    kwargs: Additional keyword arguments
        Serves the purpose of passing an index to the function, in 
        which case it will return the distance between the instance_one 
        feature at the given index and the instance_two.
    
    Returns
    -------
    float
        The euclidean distance between the two instances
    
    Notes
    -----
    It's important to note that if the keyword argument 'index' is passed 
    to the function it will expect a numeric value instead of an array as 
    the instance_two parameter.
    
    """
    # Check for kwargs
    key = 'index'
    if key in kwargs:
        index = kwargs[key]
    else:
        index = None

    one = np.array(instance_one).flatten()

    if index is not None:
        # entropy
        return np.sqrt(np.power(one[index] - instance_two, 2))

    two = np.array(instance_two)
    return np.sqrt(np.sum(np.power(np.subtract(one, two), [2 for _ in range(one.size)])))

