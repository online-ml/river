__author__ = 'Guilherme Matsumoto'

import math


def custom_distance(instance_one, instance_two, **kwargs):
    n_att1 = instance_one.size if hasattr(instance_one, 'size') else len(instance_one)
    n_att2 = instance_two.size if hasattr(instance_two, 'size') else len(instance_two)
    categorical_list = None

    if n_att1 != n_att2:
        raise ValueError("The two instances must have the same length.")

    for key, value in kwargs:
        if key == 'categorical_list':
            categorical_list = value

    print(categorical_list)

    distance_categorical = 0

    frac_numerical = n_att1/(n_att1 + len(categorical_list))

    cat_list = categorical_list

    partial_dist = 0
    for i in range(n_att1):
        if i in cat_list:
            if n_att1 != n_att2:
                distance_categorical += 1
        else:
            partial_dist += (math.pow((n_att1[i] - n_att2[i])/(n_att1[i] + n_att2[i]), 2))

    distance_numerical = math.sqrt(partial_dist) * frac_numerical
    distance_categorical = distance_categorical * (1.0 - frac_numerical)

    return distance_categorical+distance_numerical

