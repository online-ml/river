__author__ = 'Guilherme Matsumoto'

import math
import numpy as np


def mixed_distance(instance_one, instance_two, **kwargs):
    n_att1 = instance_one.size if hasattr(instance_one, 'size') else len(instance_one)
    n_att2 = instance_two.size if hasattr(instance_two, 'size') else len(instance_two)

    if n_att1 != n_att2:
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
                        return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))
            else:
                if distance_array is not None:
                    return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))\
                               / (distance_array[index]*distance_array[index]) if distance_array[index] != 0 else 0.0
                else:
                    return ((instance_one[index] - instance_two[index]) * (instance_one[index] - instance_two[index]))
        else:
            if distance_array is not None:
                return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))\
                               / (distance_array[index]*distance_array[index]) if distance_array[index] != 0 else 0.0
            else:
                return ((instance_one[index]-instance_two[index])*(instance_one[index]-instance_two[index]))

    partial_dist = []
    for i in range(n_att1):
        if categorical_list is not None:
            if i in categorical_list:
                if instance_one[i] != instance_two[i]:
                    partial_dist.append(1)
                    #print(1)
                else:
                    partial_dist.append(0)
                    #print(0)
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
    # Check for kwargs
    index = None
    for key, value in kwargs.items():
        if key == 'index':
            index = value

    one = np.array(instance_one).flatten()

    if index is not None:
        #print("entrou")
        return np.sqrt(np.power(one[index] - instance_two, 2))

    two = np.array(instance_two)
    return np.sqrt(np.sum(np.power(np.subtract(one, two), [2 for i in range(one.size)])))

if __name__ == '__main__':
    """
    inst1 = [10, 30, 1457, 2, 1]
    inst2 = [20, 50, 22321, 1, 1]
    cat_list = [3, 4]
    distance_array = [10, 20, 32000, 12312, 0]
    kwargs = {'categorical_list': cat_list, 'distance_array': distance_array}
    dist = mixed_distance(inst1, inst2, **kwargs)
    print(str(dist))
    """
    inst1 = [6, 1, 3, 9, 9]
    inst2 = [8, 2, 2, 9, 13]
    print(str(euclidean_distance(np.asarray(inst1), np.asarray(inst2))))
    print(str(np.linalg.norm(np.asarray(inst1)-np.asarray(inst2))))
