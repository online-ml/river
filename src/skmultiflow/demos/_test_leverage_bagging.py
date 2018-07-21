import warnings, logging
from skmultiflow.meta import LeverageBagging
from skmultiflow.lazy import KNN
from skmultiflow.data import SEAGenerator


def demo():
    """ _test_leverage_bagging

    This demo tests the LeverageBagging classifier on a file stream, which gives 
    instances coming from a SEA generator. 

    The test computes the performance of the LeverageBagging classifier as well 
    as the time to create the structure and classify max_samples (2000 by default) 
    instances.

    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore", ".*Passing 1d.*")
    stream = SEAGenerator(1, noise_percentage=0.067, random_state=1)
    stream.prepare_for_use()
    clf = LeverageBagging(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=1,
                          random_state=1)
    sample_count = 0
    correctly_classified = 0
    max_samples = 2000
    train_size = 200
    first = True
    if train_size > 0:
        X, y = stream.next_sample(train_size)
        clf.partial_fit(X, y, classes=stream.target_values)
        first = False

    logging.info('%s%%', 0.0)
    while sample_count < max_samples:
        if (sample_count+1) % (max_samples / 20) == 0:
            logging.info('%s%%', str(((sample_count // (max_samples / 20)+1) * 5)))
        X, y = stream.next_sample(2)
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
    print(clf.get_info())


if __name__ == '__main__':
    demo()