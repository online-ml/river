import logging
from timeit import default_timer as timer

from skmultiflow.meta import MultiOutputLearner
from skmultiflow.core import Pipeline
from skmultiflow.data import FileStream
from skmultiflow.metrics import *
from sklearn.linear_model.perceptron import Perceptron


def demo():
    """ _test_mol

    This demo tests the MOL learner on a file stream, which reads from 
    the music.csv file.

    The test computes the performance of the MOL learner as well as 
    the time to create the structure and classify all the samples in 
    the file.

    """
    # Setup logging
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Setup the file stream
    stream = FileStream("../data/datasets/music.csv", 0, 6)
    stream.prepare_for_use()

    # Setup the classifier, by default it uses Logistic Regression
    # classifier = MultiOutputLearner()
    # classifier = MultiOutputLearner(base_estimator=SGDClassifier(n_iter=100))
    classifier = MultiOutputLearner(base_estimator=Perceptron())

    # Setup the pipeline
    pipe = Pipeline([('classifier', classifier)])

    pretrain_size = 150
    logging.info('Pre training on %s samples', str(pretrain_size))
    logging.info('Total %s samples', str(stream.n_samples))
    X, y = stream.next_sample(pretrain_size)
    # classifier.fit(X, y)
    classes = stream.target_values
    classes_flat = list(set([item for sublist in classes for item in sublist]))
    pipe.partial_fit(X, y, classes=classes_flat)
    count = 0
    true_labels = []
    predicts = []
    init_time = timer()
    logging.info('Evaluating...')
    while stream.has_more_samples():
        X, y = stream.next_sample()
        # p = classifier.predict(X)
        p = pipe.predict(X)
        predicts.extend(p)
        true_labels.extend(y)
        count += 1
    perf = hamming_score(true_labels, predicts)
    logging.info('Evaluation time: %s s', str(timer() - init_time))
    logging.info('Total samples analyzed: %s', str(count))
    logging.info('The classifier\'s static Hamming score    : %0.3f' % perf)


if __name__ == '__main__':
    demo()
