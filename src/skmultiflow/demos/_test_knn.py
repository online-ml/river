from skmultiflow.lazy import KNN
from skmultiflow.data import FileStream
from timeit import default_timer as timer


def demo():
    """ _test_knn
    
    This demo tests the KNN classifier on a file stream, which gives 
    instances coming from a SEA generator. 
    
    The test computes the performance of the KNN classifier as well as 
    the time to create the structure and classify max_samples (5000 by 
    default) instances.
    
    """
    stream = FileStream('../data/datasets/sea_big.csv', -1, 1)
    stream.prepare_for_use()
    train = 200
    X, y = stream.next_sample(train)
    # t = OneHotToCategorical([[10, 11, 12, 13],
    #                         [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #                          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    # t2 = OneHotToCategorical([[10, 11, 12, 13],
    #                         [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #                          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    start = timer()
    knn = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
    # pipe = Pipeline([('one_hot_to_categorical', t), ('KNN', knn)])

    # compare = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree', leaf_size=40, metric='euclidean')

    # pipe2 = Pipeline([('one_hot_to_categorical', t2), ('KNN', compare)])

    # pipe.fit(X, y)
    # pipe2.fit(X, y)
    knn.partial_fit(X, y)
    # compare.fit(X, y)

    n_samples = 0
    max_samples = 5000
    my_corrects = 0
    # compare_corrects = 0

    while n_samples < max_samples:
        X, y = stream.next_sample()
        # my_pred = pipe.predict(X)
        my_pred = knn.predict(X)
        # compare_pred = pipe2.predict(X)
        # compare_pred = compare.predict(X)
        if y[0] == my_pred[0]:
            my_corrects += 1
        # if y[0] == compare_pred[0]:
        #     compare_corrects += 1
        n_samples += 1

    end = timer()

    print('Evaluation time: ' + str(end-start))
    print(str(n_samples) + ' samples analyzed.')
    print('My performance: ' + str(my_corrects/n_samples))
    # print('Compare performance: ' + str(compare_corrects/n_samples))


if __name__ == '__main__':
    demo()
