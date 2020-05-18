import logging
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.data import FileStream
from skmultiflow.transform import OneHotToCategorical
from sklearn.neighbors.classification import KNeighborsClassifier
from timeit import default_timer as timer


def demo():
    """ _test_knn_adwin

    This demo tests the KNNADWINClassifier on a file stream, which gives
    instances coming from a SEA generator. 
    
    The test computes the performance of the KNNADWINClassifier as well as
    the time to create the structure and classify max_samples (10000 by 
    default) instances.
    
    """
    start = timer()
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    # warnings.filterwarnings("ignore", ".*Passing 1d.*")
    stream = FileStream("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
                        "master/sea_big.csv", -1, 1)
    # stream = RandomRBFGeneratorDrift(change_speed=41.00, n_centroids=50, model_random_state=32523423,
    #                                  sample_seed=5435, n_classes=2, num_att=10, num_drift_centroids=50)

    t = OneHotToCategorical([[10, 11, 12, 13],
                            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    t2 = OneHotToCategorical([[10, 11, 12, 13],
                            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])

    knn = KNNADWINClassifier(n_neighbors=8, leaf_size=40, max_window_size=2000)
    # pipe = Pipeline([('one_hot_to_categorical', t), ('KNNClassifier', knn)])

    compare = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree', leaf_size=40, metric='euclidean')
    # pipe2 = Pipeline([('one_hot_to_categorical', t2), ('KNNClassifier', compare)])
    first = True
    train = 200
    if train > 0:
        X, y = stream.next_sample(train)
        # pipe.partial_fit(X, y, classes=stream.target_values)
        # pipe.partial_fit(X, y, classes=stream.target_values)
        # pipe2.fit(X, y)

        knn.partial_fit(X, y, classes=stream.target_values)
        compare.fit(X, y)
        first = False
    n_samples = 0
    max_samples = 10000
    my_corrects = 0
    compare_corrects = 0

    while n_samples < max_samples:
        if n_samples % (max_samples/20) == 0:
            logging.info('%s%%', str((n_samples//(max_samples/20)*5)))
        X, y = stream.next_sample()
        # my_pred = pipe.predict(X)
        my_pred = knn.predict(X)
        # my_pred = [1]
        if first:
            # pipe.partial_fit(X, y, classes=stream.target_values)
            # pipe.partial_fit(X, y, classes=stream.target_values)
            knn.partial_fit(X, y, classes=stream.target_values)
            first = False
        else:
            # pipe.partial_fit(X, y)
            knn.partial_fit(X, y)
        # compare_pred = pipe2.predict(X)
        compare_pred = compare.predict(X)
        if y[0] == my_pred[0]:
            my_corrects += 1
        if y[0] == compare_pred[0]:
            compare_corrects += 1
        n_samples += 1

    end = timer()

    print('Evaluation time: ' + str(end - start))
    print(str(n_samples) + ' samples analyzed.')
    print('My performance: ' + str(my_corrects / n_samples))
    print('Compare performance: ' + str(compare_corrects / n_samples))


if __name__ == '__main__':
    demo()
