__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption
from skmultiflow.filtering.one_hot_to_categorical import OneHotToCategorical
from skmultiflow.core.pipeline import Pipeline
from sklearn.neighbors.classification import KNeighborsClassifier


def demo():
    opt = FileOption('FILE', 'OPT_NAME', '../datasets/covtype.csv', 'csv', False)
    stream = FileStream(opt, -1, 1)
    stream.prepare_for_use()
    t = OneHotToCategorical([[10, 11, 12, 13],
                             [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    t2 = OneHotToCategorical([[10, 11, 12, 13],
                             [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]])
    knn = KNN(k=8, max_window_size=2000, leaf_size=40)
    pipe = Pipeline([('one_hot_to_categorical', t), ('KNN', knn)])

    compare = KNeighborsClassifier(n_neighbors=8, algorithm='kd_tree', leaf_size=40, metric='euclidean')

    pipe2 = Pipeline([('one_hot_to_categorical', t2), ('KNN', compare)])

    train = 2000
    X, y = stream.next_instance(train)
    pipe.fit(X, y)
    pipe2.fit(X, y)

    n_samples = 0
    max_samples = 100000
    my_corrects = 0
    compare_corrects = 0

    pass
    while n_samples < max_samples:
        X, y = stream.next_instance()
        my_pred = pipe.predict(X)
        compare_pred = pipe2.predict(X)
        if y[0] == my_pred[0]:
            my_corrects += 1
        if y[0] == compare_pred[0]:
            compare_corrects += 1
        n_samples += 1

    print(str(n_samples) + ' samples analyzed.')
    print('My performance: ' + str(my_corrects/n_samples))
    print('Compare performance: ' + str(compare_corrects/n_samples))

if __name__ == '__main__':
    demo()