import warnings, logging
from skmultiflow.meta import OzaBaggingADWINClassifier
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.data import SEAGenerator


def demo():
    """ _test_oza_bagging_adwin

    This demo tests the OzaBaggingADWINClassifier using KNNADWINClassifier as base estimator
    on samples given by a SEAGenerator. 

    The test computes the performance of the OzaBaggingADWINClassifier as well
    as the time to create the structure and classify max_samples (20000 by 
    default) instances.

    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore", ".*Passing 1d.*")
    stream = SEAGenerator(1, noise_percentage=0.067, random_state=1)

    clf = OzaBaggingADWINClassifier(base_estimator=KNNADWINClassifier(n_neighbors=8, max_window_size=2000, leaf_size=30),
                                    n_estimators=2, random_state=1)
    sample_count = 0
    correctly_classified = 0
    max_samples = 20000
    train_size = 10
    first = True
    if train_size > 0:
        X, y = stream.next_sample(train_size)
        clf.partial_fit(X, y, classes=stream.target_values)
        first = False

    while sample_count < max_samples:
        if sample_count % (max_samples/20) == 0:
            logging.info('%s%%', str((sample_count//(max_samples/20)*5)))
        X, y = stream.next_sample()
        my_pred = clf.predict(X)

        if first:
            clf.partial_fit(X, y, classes=stream.target_values)
            first = False
        else:
            clf.partial_fit(X, y)

        if my_pred is not None:
            if y[0] == my_pred[0]:
                correctly_classified += 1

        sample_count += 1

    print(str(sample_count) + ' samples analyzed.')
    print('My performance: ' + str(correctly_classified / sample_count))


if __name__ == '__main__':
    demo()